"""Generate the D->D target for the mismatch restoration map.

For every development seed and batch, encode the same mismatch-profile sentences with
D's transmitter, then pass them through the *stored* physical realization from the
paired diagnostic collection: the stored true H and standardized noise Z.

Why use the stored realization rather than regenerating from the seed?
PyTorch CPU and CUDA generators do not produce the same random sequence for the same
seed, while the original collection may have been generated on a different device.
The saved H and Z are the authoritative paired physical realization and avoid any
device-dependent RNG replay assumption.

Writes only under OUTPUT/dd_targets/.
"""
import argparse
import math
from pathlib import Path

import torch

import common as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--shared-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=cm.SEEDS)
    args = ap.parse_args()

    cm.setup_imports(Path(args.run_root) / "code")
    from utils import PowerNormalize

    torch.set_num_threads(2)

    tok = cm.load_vocab(args.shared_root)
    pad, start, end = tok["<PAD>"], tok["<START>"], tok["<END>"]
    models = cm.load_models(args.run_root, len(tok))
    d = models["D"]

    out = Path(args.output) / "dd_targets"
    out.mkdir(parents=True, exist_ok=True)

    sig = cm.snr_to_noise(cm.SNR_NOMINAL_DB)

    for seed in args.seeds:
        for b in range(args.batches):
            dest = out / "seed{}_batch_{:03d}.pt".format(seed, b)
            normal = cm.load_artifact(args.run_root, "normal", None, seed, b)
            mism = cm.load_artifact(args.run_root, "mismatch", None, seed, b)

            # Verify that the existing diagnostic collection itself is paired.
            if not torch.equal(normal["sample_ids"], mism["sample_ids"]):
                raise AssertionError(
                    "normal/mismatch sentences differ for seed {} batch {}".format(seed, b)
                )
            if not torch.equal(normal["true_h"], mism["true_h"]):
                raise AssertionError(
                    "stored normal/mismatch H differ for seed {} batch {}".format(seed, b)
                )
            if not torch.allclose(
                normal["standardized_noise"],
                mism["standardized_noise"],
                rtol=0.0,
                atol=0.0,
            ):
                raise AssertionError(
                    "stored normal/mismatch standardized noise differ for seed {} batch {}".format(seed, b)
                )

            src = mism["source_tokens"]
            mask = (src == pad).unsqueeze(-2)

            with torch.no_grad():
                # D transmitter, same source sentences.
                tx_real = PowerNormalize(d.channel_encoder(d.encoder(src, mask)))

                # Recreate the MIMO forward pass using the *stored* paired physical
                # realization rather than attempting device-dependent RNG replay.
                x = cm.to_complex(tx_real)
                x_scaled = x / math.sqrt(2.0)  # Nt=2 total-power scaling used by MIMOChannel

                h = normal["true_h"].reshape(2, 2).to(x_scaled.dtype)
                z = normal["standardized_noise"].to(x_scaled.dtype)

                faded = torch.einsum("ni,bui->bun", h, x_scaled)
                noisy = faded + sig * z

                r1_complex = cm.lmmse(noisy, h, sig)
                r1 = cm.to_real(r1_complex, tx_real.shape[1])
                r2 = d.channel_decoder(r1)
                tokens, lengths = cm.decode(d, r2, src, pad, start, end)

            if dest.exists():
                old = torch.load(dest, map_location="cpu", weights_only=False)
                if not torch.equal(old["R2_DD"], r2):
                    raise FileExistsError("refusing to overwrite differing {}".format(dest))
                continue

            torch.save(
                {
                    "seed": seed,
                    "batch_index": b,
                    "sample_ids": mism["sample_ids"],
                    "R1_DD": r1,
                    "R2_DD": r2,
                    "DD_tokens": tokens,
                    "DD_lengths": lengths,
                },
                dest,
            )
            print("saved", dest, flush=True)

    print("DD TARGETS COMPLETE")


if __name__ == "__main__":
    main()
