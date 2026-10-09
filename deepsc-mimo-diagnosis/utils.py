"""DeepSC training utilities with an explicit, traceable MIMO forward path."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from channels import ChannelTrace, MIMOChannel, MIMOConfig
from models.mutual_info import mutual_information, sample_batch


class BleuScore:
    def __init__(self, w1, w2, w3, w4):
        self.weights = (w1, w2, w3, w4)

    def compute_blue_score(self, real, predicted):
        from nltk.translate.bleu_score import sentence_bleu
        from w3lib.html import remove_tags

        scores = []
        for sent1, sent2 in zip(real, predicted):
            reference = remove_tags(sent1).split()
            hypothesis = remove_tags(sent2).split()
            scores.append(sentence_bleu([reference], hypothesis, weights=self.weights))
        return scores


class SeqtoText:
    def __init__(self, vocab_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocab_dictionary.values(), vocab_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, indices):
        words = []
        for index in indices:
            if index == self.end_idx:
                break
            words.append(self.reverse_word_map.get(index, "<UNK>"))
        return " ".join(words)


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor

    def step(self):
        self._step += 1
        rate = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()

    def rate(self, step=None):
        step = step or self._step
        return self.factor * self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)
        )


@dataclass
class PipelineTrace:
    source_tokens: torch.Tensor
    source_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    decoder_input_tokens: torch.Tensor
    target_tokens: torch.Tensor
    target_mask: torch.Tensor
    semantic_encoder_output: torch.Tensor
    channel_encoder_raw: torch.Tensor
    tx_normalized: torch.Tensor
    channel: ChannelTrace
    channel_decoder_input: torch.Tensor
    channel_decoder_output: torch.Tensor
    semantic_decoder_output: torch.Tensor
    logits: torch.Tensor
    predicted_tokens: torch.Tensor
    per_token_loss: torch.Tensor
    batch_loss: torch.Tensor
    padding_idx: int


def initNetParams(model):
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model


def subsequent_mask(size, device=None):
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool, device=device), diagonal=1)
    return mask


def create_masks(src, trg, padding_idx):
    src_mask = (src == padding_idx).unsqueeze(-2)
    trg_padding = (trg == padding_idx).unsqueeze(-2)
    look_ahead = subsequent_mask(trg.size(-1), device=trg.device)
    return src_mask, torch.logical_or(trg_padding, look_ahead)


def loss_function(logits, target, padding_idx, criterion, normalization="legacy"):
    loss = criterion(logits, target)
    mask = (target != padding_idx).type_as(loss)
    masked = loss * mask
    if normalization == "legacy":
        return masked.mean()
    if normalization == "tokens":
        return masked.sum() / mask.sum().clamp_min(1.0)
    raise ValueError("loss normalization must be legacy or tokens")


def PowerNormalize(x):
    """Preserve the original DeepSC normalization, with tensor-safe control flow."""
    rms = x.square().mean().sqrt()
    return x / torch.maximum(rms, torch.ones((), dtype=x.dtype, device=x.device))


def SNR_to_noise(snr_db):
    """Return standard deviation of each real AWGN component."""
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    return 1.0 / np.sqrt(2.0 * snr_linear)


def _channel_or_default(channel_model):
    return channel_model if channel_model is not None else MIMOChannel(MIMOConfig())


def forward_pipeline(
    model,
    src,
    trg,
    noise_std,
    pad,
    criterion,
    channel_model=None,
    collect_trace=False,
    mi_net=None,
    loss_normalization="legacy",
):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]
    src_mask, target_mask = create_masks(src, trg_inp, pad)

    semantic_encoded = model.encoder(src, src_mask)
    channel_encoded = model.channel_encoder(semantic_encoded)
    tx_normalized = PowerNormalize(channel_encoded)
    channel = _channel_or_default(channel_model)
    if collect_trace:
        received, channel_trace = channel(tx_normalized, noise_std, return_trace=True)
    else:
        received = channel(tx_normalized, noise_std, return_trace=False)
        channel_trace = None

    channel_decoded = model.channel_decoder(received)
    semantic_decoded = model.decoder(trg_inp, channel_decoded, target_mask, src_mask)
    logits = model.dense(semantic_decoded)
    vocabulary_size = logits.size(-1)
    loss = loss_function(
        logits.contiguous().view(-1, vocabulary_size),
        trg_real.contiguous().view(-1),
        pad,
        criterion,
        normalization=loss_normalization,
    )

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(tx_normalized, received)
        mi_lower_bound, _, _ = mutual_information(joint, marginal, mi_net)
        loss = loss - 0.0009 * mi_lower_bound

    if not collect_trace:
        return loss, None

    flat_loss = torch.nn.functional.cross_entropy(
        logits.contiguous().view(-1, vocabulary_size),
        trg_real.contiguous().view(-1),
        reduction="none",
        ignore_index=pad,
    ).view_as(trg_real)
    trace = PipelineTrace(
        source_tokens=src,
        source_mask=src_mask,
        sequence_lengths=(src != pad).sum(dim=1),
        decoder_input_tokens=trg_inp,
        target_tokens=trg_real,
        target_mask=target_mask,
        semantic_encoder_output=semantic_encoded,
        channel_encoder_raw=channel_encoded,
        tx_normalized=tx_normalized,
        channel=channel_trace,
        channel_decoder_input=received,
        channel_decoder_output=channel_decoded,
        semantic_decoder_output=semantic_decoded,
        logits=logits,
        predicted_tokens=logits.argmax(dim=-1),
        per_token_loss=flat_loss,
        batch_loss=loss.detach(),
        padding_idx=int(pad),
    )
    return loss, trace


def train_step(
    model, src, trg, noise_std, pad, opt, criterion, channel=None,
    mi_net=None, collect_trace=False, channel_model=None,
    loss_normalization="legacy",
):
    del channel  # retained for compatibility with the original public signature
    model.train()
    opt.zero_grad()
    loss, trace = forward_pipeline(
        model, src, trg, noise_std, pad, criterion,
        channel_model=channel_model, collect_trace=collect_trace, mi_net=mi_net,
        loss_normalization=loss_normalization,
    )
    loss.backward()
    opt.step()
    return (loss.item(), trace) if collect_trace else loss.item()


def val_step(
    model, src, trg, noise_std, pad, criterion, channel=None,
    collect_trace=False, channel_model=None, loss_normalization="legacy",
):
    del channel
    loss, trace = forward_pipeline(
        model, src, trg, noise_std, pad, criterion,
        channel_model=channel_model, collect_trace=collect_trace,
        loss_normalization=loss_normalization,
    )
    return (loss.item(), trace) if collect_trace else loss.item()


def train_mi(
    model, mi_net, src, noise_std, padding_idx, opt, channel=None,
    channel_model=None,
):
    del channel
    mi_net.train()
    opt.zero_grad()
    src_mask = (src == padding_idx).unsqueeze(-2)
    encoded = model.encoder(src, src_mask)
    transmitted = PowerNormalize(model.channel_encoder(encoded))
    received = _channel_or_default(channel_model)(transmitted, noise_std)
    joint, marginal = sample_batch(transmitted, received)
    mi_lower_bound, _, _ = mutual_information(joint, marginal, mi_net)
    loss = -mi_lower_bound
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()
    return loss.item()


def greedy_decode(
    model, src, noise_std, max_len, padding_idx, start_symbol,
    channel=None, channel_model=None,
):
    del channel
    src_mask = (src == padding_idx).unsqueeze(-2)
    semantic_encoded = model.encoder(src, src_mask)
    transmitted = PowerNormalize(model.channel_encoder(semantic_encoded))
    received = _channel_or_default(channel_model)(transmitted, noise_std)
    memory = model.channel_decoder(received)

    outputs = torch.full(
        (src.size(0), 1), start_symbol, dtype=src.dtype, device=src.device
    )
    for _ in range(max_len - 1):
        target_padding = (outputs == padding_idx).unsqueeze(-2)
        target_mask = torch.logical_or(
            target_padding, subsequent_mask(outputs.size(1), device=outputs.device)
        )
        decoded = model.decoder(outputs, memory, target_mask, src_mask)
        next_word = model.dense(decoded)[:, -1:, :].argmax(dim=-1)
        outputs = torch.cat((outputs, next_word), dim=1)
    return outputs


__all__ = [
    "BleuScore", "NoamOpt", "PipelineTrace", "PowerNormalize", "SNR_to_noise",
    "SeqtoText", "create_masks", "forward_pipeline", "greedy_decode",
    "initNetParams", "loss_function", "subsequent_mask", "train_mi",
    "train_step", "val_step",
]
