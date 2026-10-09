"""Approved semantic/physical profile definitions.

Literal corruption is preserved from the prior semantic POC.  The operation
mix is a project definition: deletion 35%, typo 30%, repeated insertion 20%,
grammar/function-word replacement 15%.
"""

import random
import string
from collections import Counter


SPECIAL = {"<PAD>", "<START>", "<END>", "<UNK>"}
CONFUSION_PAIRS = [
    ("is", "are"), ("was", "were"), ("has", "have"), ("do", "does"),
    ("this", "these"), ("that", "those"), ("a", "the"), ("in", "on"),
    ("to", "for"), ("of", "from"), ("can", "could"), ("will", "would"),
    ("should", "could"), ("its", "their"), ("he", "they"), ("she", "they"),
]
DEFAULT_OP_WEIGHTS = {
    "deletion": 0.35,
    "typo": 0.30,
    "insertion_repeat": 0.20,
    "grammar_replacement": 0.15,
}


def build_confusion_map(token_to_idx):
    result = {}
    for left, right in CONFUSION_PAIRS:
        if left in token_to_idx and right in token_to_idx:
            result[int(token_to_idx[left])] = int(token_to_idx[right])
            result[int(token_to_idx[right])] = int(token_to_idx[left])
    return result


def _weighted_choice(rng, weights):
    names = list(weights)
    values = [float(weights[name]) for name in names]
    total = sum(values)
    if total <= 0:
        raise ValueError("operation weights must sum to a positive value")
    draw = rng.random() * total
    cumulative = 0.0
    for name, weight in zip(names, values):
        cumulative += weight
        if draw <= cumulative:
            return name
    return names[-1]


def _typo_token(token, rng):
    if not token or len(token) < 3 or not any(char.isalpha() for char in token):
        return token
    operation = rng.choice(
        ["char_delete", "char_swap", "char_substitute", "char_insert"]
    )
    characters = list(token)
    alpha_positions = [
        index for index, char in enumerate(characters) if char.isalpha()
    ]
    if operation == "char_delete" and len(characters) > 3:
        del characters[rng.choice(alpha_positions)]
    elif operation == "char_swap" and len(characters) > 3:
        candidates = [
            index for index in range(len(characters) - 1)
            if characters[index].isalpha() and characters[index + 1].isalpha()
        ]
        if candidates:
            index = rng.choice(candidates)
            characters[index], characters[index + 1] = (
                characters[index + 1], characters[index]
            )
        else:
            operation = "char_substitute"
    if operation == "char_substitute":
        index = rng.choice(alpha_positions)
        original = characters[index].lower()
        alphabet = string.ascii_lowercase.replace(original, "") or string.ascii_lowercase
        replacement = rng.choice(alphabet)
        characters[index] = replacement.upper() if characters[index].isupper() else replacement
    elif operation == "char_insert":
        characters.insert(rng.choice(alpha_positions), rng.choice(string.ascii_lowercase))
    return "".join(characters)


def corrupt_sequence(
    clean_ids,
    idx_to_token,
    token_to_idx,
    rng,
    severity=0.20,
    min_total_len=7,
    max_total_len=31,
    op_weights=None,
):
    """Corrupt one unpadded sequence while preserving all special tokens."""
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")
    weights = dict(DEFAULT_OP_WEIGHTS if op_weights is None else op_weights)
    sequence = [int(value) for value in clean_ids]
    if len(sequence) < 2:
        return sequence, {"events_requested": 0, "events_applied": 0, "counts": {}}

    pad = token_to_idx["<PAD>"]
    start = token_to_idx["<START>"]
    end = token_to_idx["<END>"]
    unknown = token_to_idx["<UNK>"]
    confusion = build_confusion_map(token_to_idx)
    while sequence and sequence[-1] == pad:
        sequence.pop()
    protected = {pad, start, end, unknown}
    positions = [
        index for index, token_id in enumerate(sequence) if token_id not in protected
    ]
    original_content = len(positions)
    if original_content == 0:
        return sequence, {"events_requested": 0, "events_applied": 0, "counts": {}}

    requested = max(1, int(round(severity * original_content)))
    counts = Counter()
    typo_oov_count = 0
    for _ in range(requested):
        selected = _weighted_choice(rng, weights)
        positions = [
            index for index, token_id in enumerate(sequence)
            if token_id not in protected
        ]
        if not positions:
            break
        applied = False
        for operation in (selected, "typo", "deletion", "insertion_repeat"):
            if applied:
                break
            if operation == "deletion" and len(sequence) > min_total_len:
                del sequence[rng.choice(positions)]
                counts["deletion"] += 1
                applied = True
            elif operation == "insertion_repeat" and len(sequence) < max_total_len:
                position = rng.choice(positions)
                neighbors = [position]
                if position - 1 in positions:
                    neighbors.append(position - 1)
                if position + 1 in positions:
                    neighbors.append(position + 1)
                source_position = rng.choice(neighbors)
                sequence.insert(position + 1, int(sequence[source_position]))
                counts["insertion_repeat"] += 1
                applied = True
            elif operation == "grammar_replacement":
                eligible = [index for index in positions if int(sequence[index]) in confusion]
                if eligible:
                    position = rng.choice(eligible)
                    sequence[position] = confusion[int(sequence[position])]
                    counts["grammar_replacement"] += 1
                    applied = True
            elif operation == "typo":
                eligible = [
                    index for index in positions
                    if len(idx_to_token.get(int(sequence[index]), "")) >= 3
                    and any(char.isalpha() for char in idx_to_token[int(sequence[index])])
                ]
                if eligible:
                    position = rng.choice(eligible)
                    original = idx_to_token[int(sequence[position])]
                    changed = _typo_token(original, rng)
                    if changed != original:
                        new_id = int(token_to_idx.get(changed, unknown))
                        typo_oov_count += int(new_id == unknown and original != "<UNK>")
                        sequence[position] = new_id
                        counts["typo"] += 1
                        applied = True

    if sequence and sequence[0] != start and start in clean_ids:
        sequence.insert(0, start)
    if end in clean_ids:
        if end in sequence:
            sequence = sequence[:sequence.index(end) + 1]
        else:
            if len(sequence) >= max_total_len:
                sequence = sequence[:max_total_len - 1]
            sequence.append(end)
    if len(sequence) > max_total_len:
        sequence = (
            sequence[:max_total_len - 1] + [end]
            if sequence and sequence[-1] == end else sequence[:max_total_len]
        )
    metadata = {
        "severity_requested": float(severity),
        "original_total_length": int(len(clean_ids)),
        "corrupted_total_length": int(len(sequence)),
        "original_content_tokens": int(original_content),
        "unk_tokens_protected_from_editing": True,
        "events_requested": int(requested),
        "events_applied": int(sum(counts.values())),
        "counts": dict(counts),
        "typo_to_unk_count": int(typo_oov_count),
    }
    return sequence, metadata


PROFILE_SPECS = {
    "normal": {"label": "normal", "snr_db": 10.0},
    "low_snr": {"label": "low_snr", "snr_db": 1.0},
    "high_snr_control": {"label": "control", "snr_db": 19.0},
    "literal": {"label": "literal", "snr_db": 10.0, "severity": 0.20},
    "mismatch": {"label": "mismatch", "snr_db": 10.0},
    "ood_hansard": {"label": "ood", "snr_db": 10.0},
    "ood_arxiv": {"label": "HOLDOUT", "snr_db": 10.0},
}


__all__ = [
    "CONFUSION_PAIRS", "DEFAULT_OP_WEIGHTS", "PROFILE_SPECS",
    "build_confusion_map", "corrupt_sequence",
]
