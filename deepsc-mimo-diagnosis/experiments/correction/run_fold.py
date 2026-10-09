"""One leave-one-physical-seed-out fold of the restoration experiments (correction_v1).

Stage 1  oracle cause: does each restoration map G_c restore its cause? (representation NMSE /
         cosine at the actuation stage, BLEU-1/BLEU-4/token accuracy after the frozen receiver)
Stage 1b nominal harm and wrong-map harm
Stage 2  routing with the diagnostic's held-out probabilities: hard (sentence / block),
         soft same-stage rule (sentence / block)

Train data: the four training seeds only. Physical maps are trained on channels re-simulated
over training-seed symbols (exact LMMSE physics of channels.py); affine maps on stored pairs.
Severity generalization: CSI maps see rho in {0.99, 0.90} only; correlation maps see
rho_s in {0.3, 0.9} only. Writes only under OUTPUT/folds/.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import common as cm
from restoration_maps import AffineMap, BlockMatrixNet, TokenMLP

TRAIN_SEV = {"imperfect_csi": [0.99, 0.90], "spatial_correlation": [0.3, 0.9]}
PHYSICAL = ["low_snr", "imperfect_csi", "spatial_correlation"]
# pre-declared main restoration map per cause (used for routing and wrong-map tests)
MAIN = {"low_snr": "mlp", "imperfect_csi": "block_matrix", "spatial_correlation": "mlp",
        "mismatch": "affine_R2"}


class Ctx:
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def slim(a):
    keep = ["profile", "severity", "snr_db", "physical_seed", "batch_index", "sample_ids",
            "source_tokens", "clean_reference_tokens", "R1_equalized_decoder_input",
            "R2_channel_decoder_output", "R3_generated_tokens", "R3_generated_lengths",
            "true_h", "estimated_h", "underlying_iid_h", "faded_signal", "R0_noisy_received"]
    return {k: a[k] for k in keep if k in a}


class Store:
    def __init__(self, run_root, out_root):
        self.run_root, self.out_root, self.cache = run_root, out_root, {}

    def get(self, profile, severity, seed, batch):
        key = (profile, severity, seed, batch)
        if key not in self.cache:
            if len(self.cache) > 400:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = slim(cm.load_artifact(self.run_root, profile, severity, seed, batch))
        return self.cache[key]

    def dd(self, seed, batch):
        key = ("DD", seed, batch)
        if key not in self.cache:
            self.cache[key] = torch.load(Path(self.out_root) / "dd_targets" /
                                         "seed{}_batch_{:03d}.pt".format(seed, batch),
                                         map_location="cpu", weights_only=False)
        return self.cache[key]


# ------------------------------------------------------------------ channel re-simulation
def cn(shape, gen, std=1.0):
    return torch.complex(torch.randn(shape, generator=gen) * std, torch.randn(shape, generator=gen) * std)


def kron_root(rho):
    r = torch.tensor([[1.0, rho], [rho, 1.0]])
    w, v = torch.linalg.eigh(r)
    return ((v * w.clamp_min(0).sqrt()) @ v.T).to(torch.complex64)


def simulate(cause, x, umask, length, gen, sig_n, sig_l):
    """Paired (input, target) equalized blocks for one cause; returns complex input, real input,
    real target and receiver-available context."""
    h = cn((2, 2), gen, 1 / math.sqrt(2))
    z = cn(x.shape, gen)                                  # standardized complex noise
    if cause == "imperfect_csi":
        rho = TRAIN_SEV[cause][int(torch.randint(len(TRAIN_SEV[cause]), (1,), generator=gen))]
        h_hat = rho * h + math.sqrt(1 - rho ** 2) * cn((2, 2), gen, 1 / math.sqrt(2))
        y = torch.einsum("ni,bui->bun", h, x) + sig_n * z
        inp, tgt, h_rx, sig = cm.lmmse(y, h_hat, sig_n), cm.lmmse(y, h, sig_n), h_hat, sig_n
    elif cause == "low_snr":
        hx = torch.einsum("ni,bui->bun", h, x)
        inp, tgt, h_rx, sig = cm.lmmse(hx + sig_l * z, h, sig_l), cm.lmmse(hx + sig_n * z, h, sig_n), h, sig_l
    elif cause == "spatial_correlation":
        rho = TRAIN_SEV[cause][int(torch.randint(len(TRAIN_SEV[cause]), (1,), generator=gen))]
        s = kron_root(rho)
        hc = s @ h @ s
        inp = cm.lmmse(torch.einsum("ni,bui->bun", hc, x) + sig_n * z, hc, sig_n)
        tgt = cm.lmmse(torch.einsum("ni,bui->bun", h, x) + sig_n * z, h, sig_n)
        h_rx, sig = hc, sig_n
    else:                                                 # nominal (identity examples)
        inp = cm.lmmse(torch.einsum("ni,bui->bun", h, x) + sig_n * z, h, sig_n)
        tgt, h_rx, sig = inp, h, sig_n
    ctx = cm.context_vector(inp, umask, h_rx, sig)
    return inp, cm.to_real(inp, length), cm.to_real(tgt, length), ctx


def masked_mse(pred, tgt, mask):
    m = mask[..., None].float()
    return ((pred - tgt) ** 2 * m).sum() / (m.sum() * pred.shape[-1])


def train_physical(cause, kind, pool, c, steps):
    gen = torch.Generator().manual_seed(1000 + PHYSICAL.index(cause) * 10 + (kind == "block_matrix"))
    model = BlockMatrixNet(17) if kind == "block_matrix" else TokenMLP(16, 17, variance=(cause != "imperfect_csi"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        loss = 0.0
        for j in range(5):                                # 4 impaired blocks + 1 identity block (20%)
            x, tmask, length = pool[int(torch.randint(len(pool), (1,), generator=gen))]
            # simulate the whole block so the block context has the same statistics as at test
            # time (128 sentences); the loss is computed on a random 32-sentence subset for speed
            which = cause if j < 4 else "nominal"
            inp_c, inp, tgt, ctx = simulate(which, x, cm.use_mask(tmask), length, gen, c.sig_n, c.sig_l)
            rows = torch.randperm(x.shape[0], generator=gen)[:32]
            inp_c, inp, tgt, tm = inp_c[rows], inp[rows], tgt[rows], tmask[rows]
            if kind == "block_matrix":
                pred = cm.to_real(model(inp_c, ctx[None]), length)
                loss = loss + masked_mse(pred, tgt, tm)
            else:
                pred, logvar = model(inp, ctx[None].expand(inp.shape[0], -1))
                loss = loss + masked_mse(pred, tgt, tm)
                if logvar is not None:
                    m = tm[..., None].float()
                    nll = 0.5 * (logvar + (tgt - pred.detach()) ** 2 / logvar.exp())
                    loss = loss + (nll * m).sum() / (m.sum() * 16)
        opt.zero_grad()
        (loss / 5).backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0:
            log("  {} {} step {} loss {:.5f}".format(cause, kind, step, float(loss / 5)))
    model.eval()
    return model


def train_mlp_r2(X, T, N, steps, gen_seed=77):
    """Residual MLP on R2 tokens for mismatch, trained on stored pairs + identity on D->D."""
    gen = torch.Generator().manual_seed(gen_seed)
    model = TokenMLP(128, 0, hidden=512)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        i = torch.randint(len(X), (3277,), generator=gen)
        k = torch.randint(len(N), (819,), generator=gen)  # 20% identity samples
        inp = torch.cat([X[i], N[k]])[None]
        tgt = torch.cat([T[i], N[k]])[None]
        pred, _ = model(inp, None)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0:
            log("  mismatch mlp_R2 step {} loss {:.5f}".format(step, float(loss)))
    model.eval()
    return model


# ------------------------------------------------------------------ applying maps
def block_sigma(a):
    """Noise std the receiver's own LMMSE equalizer used for this block (channels.py equalizes
    with the actual noise variance). The sanity check below verifies that equalizing R0 with this
    value reproduces the stored R1 exactly, i.e. it is the receiver-side quantity already required
    by equalization, not extra information. The diagnostic never sees it."""
    return cm.snr_to_noise(float(a["snr_db"]))


def receiver(c, a):
    return c.models["D"] if a["profile"] == "mismatch" else c.models["C"]


def corrected_r1(c, cause, kind, a):
    """Restored R1 (real [B, L, 16]) for a physical map applied to block a."""
    r1 = a["R1_equalized_decoder_input"]
    length = r1.shape[1]
    tm = cm.token_mask(a, c.pad)
    rc = cm.to_complex(r1)
    sig = block_sigma(a)
    with torch.no_grad():
        if kind == "affine":
            return c.maps[(cause, "affine")](r1)
        if kind == "oracle_matrix":                      # uses the true H: upper bound only
            m = cm.lmmse_matrix(a["true_h"], sig) @ torch.linalg.inv(cm.lmmse_matrix(a["estimated_h"], sig))
            return cm.to_real(torch.einsum("ni,bui->bun", m.to(rc.dtype), rc), length)
        ctx = cm.context_vector(rc, cm.use_mask(tm), a["estimated_h"], sig)
        model = c.maps[(cause, kind)]
        if kind == "block_matrix":
            return cm.to_real(model(rc, ctx[None]), length)
        mean, logvar = model(r1, ctx[None].expand(r1.shape[0], -1))
        return mean


def corrected_r2(c, cause, kind, a):
    """Restored R2 of block a, in the coordinates of block a's receiver."""
    rx = receiver(c, a)
    with torch.no_grad():
        if cause == "mismatch":
            if kind == "affine_R2":
                return c.maps[("mismatch", "affine_R2")](a["R2_channel_decoder_output"])
            if kind == "mlp_R2":
                return c.maps[("mismatch", "mlp_R2")](a["R2_channel_decoder_output"], None)[0]
            if kind == "affine_R1":
                return rx.channel_decoder(c.maps[("mismatch", "affine_R1")](a["R1_equalized_decoder_input"]))
        return rx.channel_decoder(corrected_r1(c, cause, kind, a))


def decode_metrics(c, rx, r2, a):
    tokens, lengths = cm.decode(rx, r2, a["source_tokens"], c.pad, c.start, c.end)
    return cm.text_metrics(a["clean_reference_tokens"], tokens, lengths, c.special, c.end), (tokens, lengths)


def stored_metrics(c, a, tokens=None, lengths=None):
    tokens = a["R3_generated_tokens"] if tokens is None else tokens
    lengths = a["R3_generated_lengths"] if lengths is None else lengths
    return cm.text_metrics(a["clean_reference_tokens"], tokens, lengths, c.special, c.end)


def summarize_text(m):
    b1, b4, acc = m
    return {"bleu1": float(b1.mean()), "bleu4": float(b4.mean()), "token_acc": float(acc.mean()), "n": int(len(b1))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--shared-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--held-out-seed", type=int, required=True)
    ap.add_argument("--probs-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--mlp-r2-steps", type=int, default=2000)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--wrong-map-batches", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cm.set_seed(20261007 + args.held_out_seed)
    cm.setup_imports(Path(args.run_root) / "code")
    if args.smoke:
        args.steps, args.mlp_r2_steps, args.batches, args.wrong_map_batches = 30, 30, 2, 1

    c = Ctx()
    tok = cm.load_vocab(args.shared_root)
    c.pad, c.start, c.end = tok["<PAD>"], tok["<START>"], tok["<END>"]
    c.special = {c.pad, c.start, c.end}
    c.models = cm.load_models(args.run_root, len(tok))
    c.sig_n, c.sig_l = cm.snr_to_noise(cm.SNR_NOMINAL_DB), cm.snr_to_noise(cm.SNR_LOW_DB)
    store = Store(args.run_root, args.output)
    held = args.held_out_seed
    train_seeds = [s for s in cm.SEEDS if s != held]
    batches = list(range(args.batches))
    train_batches = list(range(10)) if not args.smoke else [0, 1]
    out_dir = Path(args.output) / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"held_out_seed": held, "train_seeds": train_seeds, "train_severities": TRAIN_SEV,
              "main_maps": MAIN, "steps": args.steps, "smoke": args.smoke,
              "target_space": "physical maps: receiver C equalized R1 under the paired nominal event; "
                              "mismatch: receiver D's own R2 (D->D)"}

    # ---- sanity checks on one stored batch (pairing and reconstruction)
    a_n = store.get("normal", None, held, 0)
    sig = block_sigma(a_n)
    r1_rec = cm.to_real(cm.lmmse(a_n["R0_noisy_received"], a_n["estimated_h"], sig), a_n["R1_equalized_decoder_input"].shape[1])
    x = cm.transmitted_symbols(a_n)
    fad = torch.einsum("ni,bui->bun", a_n["true_h"].reshape(2, 2), x)
    (tk, ln) = cm.decode(c.models["C"], a_n["R2_channel_decoder_output"], a_n["source_tokens"], c.pad, c.start, c.end)
    same_tokens = float((tk[:, :a_n["R3_generated_tokens"].shape[1]] == a_n["R3_generated_tokens"][:, :tk.shape[1]]).all(1).float().mean())
    a_l = store.get("low_snr", None, held, 0)
    r1_low = cm.to_real(cm.lmmse(a_l["R0_noisy_received"], a_l["estimated_h"], block_sigma(a_l)), a_l["R1_equalized_decoder_input"].shape[1])
    result["sanity"] = {
        "r1_reconstruction_max_abs_err": float((r1_rec - a_n["R1_equalized_decoder_input"]).abs().max()),
        "r1_reconstruction_low_snr_max_abs_err": float((r1_low - a_l["R1_equalized_decoder_input"]).abs().max()),
        "faded_reconstruction_rel_err": float((fad - a_n["faded_signal"]).abs().max() / a_n["faded_signal"].abs().max()),
        "redecode_identical_fraction": same_tokens,
    }
    log("sanity", result["sanity"])

    # ---- training data
    log("building training pool")
    pool = []
    for s in train_seeds:
        for b in train_batches:
            a = store.get("normal", None, s, b)
            pool.append((cm.transmitted_symbols(a), cm.token_mask(a, c.pad), a["R1_equalized_decoder_input"].shape[1]))
    c.maps = {}

    def stacked(profile, sevs, field, seeds):
        X, T, N = [], [], []
        for s in seeds:
            for b in train_batches:
                n = store.get("normal", None, s, b)
                m = cm.token_mask(n, c.pad)
                N.append(n["R1_equalized_decoder_input"][m])
                for sev in sevs:
                    a = store.get(profile, sev, s, b)
                    X.append(a[field][m])
                    T.append(n["R1_equalized_decoder_input"][m])
        return torch.cat(X), torch.cat(T), torch.cat(N)

    for cause in PHYSICAL:
        sevs = TRAIN_SEV.get(cause, [None])
        X, T, N = stacked(cause, sevs, "R1_equalized_decoder_input", train_seeds)
        c.maps[(cause, "affine")] = AffineMap(16).fit(X, T, N)
        log("fitted affine", cause, X.shape)
        c.maps[(cause, "mlp")] = train_physical(cause, "mlp", pool, c, args.steps)
        if cause == "imperfect_csi":
            c.maps[(cause, "block_matrix")] = train_physical(cause, "block_matrix", pool, c, args.steps)

    Xr2, Tr2, Xr1, Tr1 = [], [], [], []
    for s in train_seeds:
        for b in train_batches:
            a, dd = store.get("mismatch", None, s, b), store.dd(s, b)
            m = cm.token_mask(a, c.pad)
            Xr2.append(a["R2_channel_decoder_output"][m]); Tr2.append(dd["R2_DD"][m])
            Xr1.append(a["R1_equalized_decoder_input"][m]); Tr1.append(dd["R1_DD"][m])
    Xr2, Tr2, Xr1, Tr1 = map(torch.cat, (Xr2, Tr2, Xr1, Tr1))
    c.maps[("mismatch", "affine_R2")] = AffineMap(128).fit(Xr2, Tr2, Tr2)
    c.maps[("mismatch", "affine_R1")] = AffineMap(16).fit(Xr1, Tr1, Tr1)
    c.maps[("mismatch", "mlp_R2")] = train_mlp_r2(Xr2, Tr2, Tr2, args.mlp_r2_steps)
    log("mismatch maps fitted", Xr2.shape)

    # ---- stage 1: oracle-cause restoration
    methods = {"low_snr": ["affine", "mlp"], "imperfect_csi": ["affine", "mlp", "block_matrix", "oracle_matrix"],
               "spatial_correlation": ["affine", "mlp"], "mismatch": ["affine_R2", "affine_R1", "mlp_R2"]}
    oracle_rows, varcal = [], []
    for cause, kinds in methods.items():
        sevs = cm.PROFILE_DIRS[cause]
        for _, sev in sevs:
            for b in batches:
                a = store.get(cause, sev, held, b)
                n = store.get("normal", None, held, b)
                tm = cm.token_mask(a, c.pad)
                rx = receiver(c, a)
                if cause == "mismatch":
                    dd = store.dd(held, b)
                    target_r2, ref_tokens = dd["R2_DD"], (dd["DD_tokens"], dd["DD_lengths"])
                    target_stage, inp_stage = dd["R2_DD"], a["R2_channel_decoder_output"]
                else:
                    target_r2, ref_tokens = n["R2_channel_decoder_output"], (n["R3_generated_tokens"], n["R3_generated_lengths"])
                    target_stage, inp_stage = n["R1_equalized_decoder_input"], a["R1_equalized_decoder_input"]
                base = summarize_text(stored_metrics(c, a))
                nominal = summarize_text(stored_metrics(c, a, *ref_tokens))
                nm, cs = cm.rep_metrics(inp_stage, target_stage, tm)
                oracle_rows.append(dict(cause=cause, severity=sev, batch=b, method="none", nmse=nm, cos=cs,
                                        nominal_bleu1=nominal["bleu1"], **base))
                for kind in kinds:
                    if cause == "mismatch":
                        r2 = corrected_r2(c, cause, kind, a)
                        if kind == "affine_R1":
                            nm, cs = cm.rep_metrics(c.maps[("mismatch", "affine_R1")](a["R1_equalized_decoder_input"]),
                                                    store.dd(held, b)["R1_DD"], tm)
                        else:
                            nm, cs = cm.rep_metrics(r2, target_stage, tm)
                    else:
                        r1c = corrected_r1(c, cause, kind, a)
                        nm, cs = cm.rep_metrics(r1c, target_stage, tm)
                        with torch.no_grad():
                            r2 = rx.channel_decoder(r1c)
                            if kind == "mlp" and cause != "imperfect_csi":
                                rcx = cm.to_complex(a["R1_equalized_decoder_input"])
                                ctx = cm.context_vector(rcx, cm.use_mask(tm), a["estimated_h"], block_sigma(a))
                                mean, logvar = c.maps[(cause, "mlp")](a["R1_equalized_decoder_input"], ctx[None].expand(tm.shape[0], -1))
                                err = ((mean - target_stage) ** 2)[tm].sum(-1)
                                var = logvar.exp()[tm].sum(-1)
                                rank = lambda v: torch.argsort(torch.argsort(v)).float()
                                rho = float(np.corrcoef(rank(err).numpy(), rank(var).numpy())[0, 1])
                                varcal.append({"cause": cause, "severity": sev, "batch": b, "spearman_var_vs_sqerr": rho})
                    r2_nm, _ = cm.rep_metrics(r2, target_r2, tm)
                    met, _ = decode_metrics(c, rx, r2, a)
                    oracle_rows.append(dict(cause=cause, severity=sev, batch=b, method=kind, nmse=nm, cos=cs,
                                            r2_nmse=r2_nm, nominal_bleu1=nominal["bleu1"], **summarize_text(met)))
                log("oracle", cause, sev, b, [(r["method"], round(r["bleu1"], 3)) for r in oracle_rows[-(len(kinds) + 1):]])
    result["oracle"] = oracle_rows
    result["variance_calibration"] = varcal

    # ---- stage 1b: nominal harm (apply each map to nominal blocks of its own receiver)
    harm = []
    for b in batches:
        n = store.get("normal", None, held, b)
        n_ref = summarize_text(stored_metrics(c, n))
        for cause in PHYSICAL:
            for kind in methods[cause]:
                if kind == "oracle_matrix":
                    continue
                with torch.no_grad():
                    r2 = c.models["C"].channel_decoder(corrected_r1(c, cause, kind, n))
                met, _ = decode_metrics(c, c.models["C"], r2, n)
                harm.append(dict(map=cause, method=kind, batch=b, delta_bleu1=summarize_text(met)["bleu1"] - n_ref["bleu1"]))
        dd = store.dd(held, b)
        dd_ref = summarize_text(stored_metrics(c, n, dd["DD_tokens"], dd["DD_lengths"]))
        fake = dict(n, profile="mismatch", R2_channel_decoder_output=dd["R2_DD"], R1_equalized_decoder_input=dd["R1_DD"])
        for kind in methods["mismatch"]:
            r2 = corrected_r2(c, "mismatch", kind, fake)
            met, _ = decode_metrics(c, c.models["D"], r2, fake)
            harm.append(dict(map="mismatch", method=kind, batch=b, delta_bleu1=summarize_text(met)["bleu1"] - dd_ref["bleu1"]))
        log("nominal harm batch", b)
    result["nominal_harm"] = harm

    # ---- stage 1c: wrong-map harm (main map of cause c' applied to blocks of cause c != c')
    wrong = []
    causes = PHYSICAL + ["mismatch"]
    for true_cause in causes:
        for _, sev in cm.PROFILE_DIRS[true_cause]:
            for b in batches[:args.wrong_map_batches]:
                a = store.get(true_cause, sev, held, b)
                rx = receiver(c, a)
                base = summarize_text(stored_metrics(c, a))
                for applied in causes:
                    if applied == true_cause:
                        continue
                    r2 = corrected_r2(c, applied, MAIN[applied], a)
                    met, _ = decode_metrics(c, rx, r2, a)
                    wrong.append(dict(true_cause=true_cause, severity=sev, applied_map=applied, batch=b,
                                      delta_bleu1=summarize_text(met)["bleu1"] - base["bleu1"]))
    result["wrong_map"] = wrong
    log("wrong-map done")

    # ---- stage 2: routing with the diagnostic's held-out probabilities
    probs = {}
    def read_probs(seed):
        table = {}
        with open(Path(args.probs_dir) / "fusion_probs_seed{}.csv".format(seed)) as h:
            for r in csv.DictReader(h):
                rel = r["path"].split("/traces/development/")[1]
                table[(rel, int(r["index"]))] = np.array([float(r["p_" + k]) for k in cm.LABELS])
        return table
    probs = read_probs(held)
    others = np.concatenate([np.stack(list(read_probs(s).values())) for s in train_seeds])
    other_labels = None
    # temperature fitted on the other folds' held-out probabilities (NLL), never on this fold
    lab_rows = []
    for s in train_seeds:
        with open(Path(args.probs_dir) / "fusion_probs_seed{}.csv".format(s)) as h:
            for r in csv.DictReader(h):
                lab_rows.append(cm.LABELS.index(r["label"]))
    other_labels = np.array(lab_rows)
    logp = np.log(np.clip(others, 1e-12, 1))
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 6.0, 45):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(z)), other_labels] - np.log(np.exp(z).sum(1))).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    result["temperature"] = best_t

    def calibrated(p):
        z = np.log(np.clip(p, 1e-12, 1)) / best_t
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    route_rows = []
    for profile, runs in cm.PROFILE_DIRS.items():
        for prof, sev in runs:
            for b in batches:
                a = store.get(prof, sev, held, b)
                rel = str(cm.artifact_path(args.run_root, prof, sev, held, b)).split("/traces/development/")[1]
                P = calibrated(np.stack([probs[(rel, i)] for i in range(a["sample_ids"].shape[0])]))
                rx = receiver(c, a)
                is_d = a["profile"] == "mismatch"
                avail = ["mismatch"] if is_d else PHYSICAL      # maps defined for this receiver
                r2_none = a["R2_channel_decoder_output"]
                cand = {k: corrected_r2(c, k, MAIN[k], a) for k in avail}
                pb = P.mean(0)
                true = cm.PROFILE_LABEL[profile]
                variants = {}
                variants["oracle"] = cand.get(true, r2_none)
                hs = P.argmax(1)
                r2 = r2_none.clone()
                for i in range(len(hs)):
                    k = cm.LABELS[hs[i]]
                    if k in cand:
                        r2[i] = cand[k][i]
                variants["hard_sentence"] = r2
                kb = cm.LABELS[int(pb.argmax())]
                variants["hard_block"] = cand.get(kb, r2_none)
                with torch.no_grad():
                    if is_d:
                        w_b = float(pb[cm.LABELS.index("mismatch")])
                        variants["soft_block"] = r2_none + w_b * (cand["mismatch"] - r2_none)
                        w_s = torch.tensor(P[:, cm.LABELS.index("mismatch")], dtype=torch.float32)[:, None, None]
                        variants["soft_sentence"] = r2_none + w_s * (cand["mismatch"] - r2_none)
                    else:
                        r1 = a["R1_equalized_decoder_input"]
                        r1c = {k: corrected_r1(c, k, MAIN[k], a) for k in PHYSICAL}
                        for name, weights in (("soft_block", np.tile(pb, (len(P), 1))), ("soft_sentence", P)):
                            w = {k: torch.tensor(weights[:, cm.LABELS.index(k)], dtype=torch.float32)[:, None, None] for k in PHYSICAL}
                            mix = (1 - sum(w.values())) * r1 + sum(w[k] * r1c[k] for k in PHYSICAL)
                            variants[name] = rx.channel_decoder(mix)
                base = summarize_text(stored_metrics(c, a))
                row = dict(profile=profile, severity=sev, batch=b, true=true, none=base["bleu1"],
                           block_prediction=kb, sentence_corrected_fraction=float(np.mean([cm.LABELS[h] in cand for h in hs])))
                for name, r2v in variants.items():
                    if torch.equal(r2v, r2_none):
                        row[name] = base["bleu1"]
                    else:
                        row[name] = summarize_text(decode_metrics(c, rx, r2v, a)[0])["bleu1"]
                route_rows.append(row)
                log("route", profile, sev, b, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
    result["routing"] = route_rows

    dest = out_dir / "fold_seed{}{}.json".format(held, "_smoke" if args.smoke else "")
    dest.write_text(json.dumps(result, indent=1, default=float))
    log("FOLD COMPLETE", dest)


if __name__ == "__main__":
    main()
