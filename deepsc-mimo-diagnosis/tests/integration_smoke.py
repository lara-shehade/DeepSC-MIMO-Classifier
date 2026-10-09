"""Run training -> checkpoint -> collection -> feature extraction on synthetic data."""

import csv
import json
from pathlib import Path
import pickle
import tempfile

from analysis.extract_features import artifact_row
from analysis.extract_sentence_features import artifact_rows as sentence_artifact_rows
from analysis.first_divergence import main as first_divergence_main
from analysis.neural_diagnostics import make_model
from collect_profiles import main as collect_profiles_main
from main import main
import torch


def write_fixture(root):
    token_to_idx = {
        "<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3,
        "hello": 4, "world": 5, "mimo": 6, "semantic": 7,
    }
    (root / "vocab.json").write_text(
        json.dumps({"token_to_idx": token_to_idx}), encoding="utf-8"
    )
    samples = [
        [1, 4, 5, 2], [1, 6, 7, 2], [1, 4, 6, 5, 2], [1, 7, 4, 2],
    ]
    for split in ("train", "test"):
        with (root / "{}_data.pkl".format(split)).open("wb") as handle:
            pickle.dump(samples, handle)


def main_smoke():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_fixture(root)
        checkpoint_dir = root / "checkpoints"
        common = [
            "--data-root", str(root),
            "--vocab-file", str(root / "vocab.json"),
            "--allow-nonshared-vocab",
            "--device", "cpu",
            "--batch-size", "2",
            "--d-model", "8",
            "--dff", "16",
            "--num-layers", "1",
            "--num-heads", "2",
            "--fading-scope", "sample",
            "--seed", "3",
            "--channel-seed", "5",
        ]
        result = main(common + [
            "--epochs", "1",
            "--max-train-batches", "1",
            "--max-validation-batches", "1",
            "--checkpoint-path", str(checkpoint_dir),
        ])
        assert result == 0 and (checkpoint_dir / "best.pth").is_file()
        assert (checkpoint_dir / "last.pth").is_file()
        assert {path.name for path in checkpoint_dir.glob("*.pth")} == {
            "best.pth", "last.pth"
        }
        try:
            main(common + [
                "--epochs", "1",
                "--max-train-batches", "1",
                "--max-validation-batches", "1",
                "--checkpoint-path", str(checkpoint_dir),
            ])
        except FileExistsError:
            pass
        else:
            raise AssertionError("non-empty checkpoint directory was not rejected")

        result = main(common + [
            "--epochs", "2",
            "--max-train-batches", "1",
            "--max-validation-batches", "1",
            "--checkpoint-path", str(checkpoint_dir),
            "--resume-checkpoint", str(checkpoint_dir / "last.pth"),
        ])
        assert result == 0
        with (checkpoint_dir / "history.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 2

        # A stop/resume boundary must be bit-exact against uninterrupted training.
        uninterrupted = root / "uninterrupted"
        result = main(common + [
            "--epochs", "2",
            "--max-train-batches", "1",
            "--max-validation-batches", "1",
            "--checkpoint-path", str(uninterrupted),
        ])
        assert result == 0
        resumed_state = torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=False)
        uninterrupted_state = torch.load(uninterrupted / "last.pth", map_location="cpu", weights_only=False)
        for name, value in resumed_state["model_state_dict"].items():
            assert torch.equal(value, uninterrupted_state["model_state_dict"][name]), name

        # Approved schema-v3 collector exercises target-free R3 on the same model.
        shared = root / "shared"
        for domain in ("europarl", "ukhansard", "arxiv"):
            domain_root = shared / domain
            domain_root.mkdir(parents=True)
            samples = [
                [1, 4, 5, 2], [1, 6, 7, 2], [1, 4, 6, 5, 2], [1, 7, 4, 2],
            ] * 3
            for split in ("train", "test"):
                with (domain_root / "{}_data.pkl".format(split)).open("wb") as handle:
                    pickle.dump(samples, handle)
        profile_root = root / "profile_traces"
        result = collect_profiles_main([
            "--profile", "normal", "--physical-seed", "3101", "--seed-index", "0",
            "--shared-data-root", str(shared), "--vocab-file", str(root / "vocab.json"),
            "--checkpoint-c", str(checkpoint_dir / "last.pth"),
            "--checkpoint-d", str(checkpoint_dir / "last.pth"),
            "--output-root", str(profile_root), "--device", "cpu",
            "--batch-size", "2", "--batches", "1", "--max-generation-length", "6",
            "--allow-nonshared-vocab", "--allow-incomplete-checkpoint",
        ])
        assert result == 0
        profile_artifact = next(profile_root.rglob("artifacts/*.pt"))
        profile_payload = torch.load(profile_artifact, map_location="cpu", weights_only=False)
        assert profile_payload["schema_version"] == 3
        assert profile_payload["R0_noisy_received"].is_complex()
        assert profile_payload["R3_free_running_decoder_hidden"].shape[-1] == 8
        assert profile_payload["R3_generated_mask"].dtype == torch.bool
        sentence_rows = sentence_artifact_rows(profile_artifact)
        assert len(sentence_rows) == 2
        assert "R0__antenna_corr_abs" in sentence_rows[0]
        assert "R3__effective_rank" in sentence_rows[0]
        for architecture, stage in (
            ("cnn", "R0"), ("tcn", "R1"),
            ("conv_transformer", "R2"), ("fusion", "fusion"),
        ):
            diagnostic = make_model(architecture, stage)
            stages = ("R0", "R1", "R2", "R3") if stage == "fusion" else (stage,)
            values = {
                name: torch.randn(2, 7, {"R0": 4, "R1": 16, "R2": 128, "R3": 128}[name])
                for name in stages
            }
            masks = {name: torch.ones(2, 7, dtype=torch.bool) for name in stages}
            assert diagnostic(values, masks).shape == (2, 7)

        checkpoint_d = root / "D_last.pth"
        d_payload = torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=False)
        d_payload["model_state_dict"]["channel_decoder.linear1.weight"] = (
            d_payload["model_state_dict"]["channel_decoder.linear1.weight"] + 0.01
        )
        torch.save(d_payload, checkpoint_d)
        base_collection = [
            "--physical-seed", "3101", "--seed-index", "0",
            "--shared-data-root", str(shared), "--vocab-file", str(root / "vocab.json"),
            "--checkpoint-c", str(checkpoint_dir / "last.pth"),
            "--checkpoint-d", str(checkpoint_d),
            "--output-root", str(profile_root), "--device", "cpu",
            "--batch-size", "2", "--batches", "1", "--max-generation-length", "6",
            "--allow-nonshared-vocab", "--allow-incomplete-checkpoint",
        ]
        for profile, severity in (
            ("low_snr", None), ("imperfect_csi", "0.90"),
            ("spatial_correlation", "0.3"), ("mismatch", None),
        ):
            arguments = ["--profile", profile] + base_collection
            if severity is not None:
                arguments += ["--severity", severity]
            assert collect_profiles_main(arguments) == 0
        divergence_report = root / "first_divergence.json"
        assert first_divergence_main([
            str(profile_root), "--output", str(divergence_report)
        ]) == 0
        assert divergence_report.is_file()

        output_root = root / "traces"
        result = main(common + [
            "--collect-only",
            "--load-checkpoint", str(checkpoint_dir / "best.pth"),
            "--max-collected-batches", "1",
            "--output-root", str(output_root),
            "--run-id", "smoke",
        ])
        assert result == 0
        run_dir = output_root / "baseline" / "smoke"
        artifacts = list((run_dir / "artifacts").glob("*.pt"))
        assert len(artifacts) == 1
        row = artifact_row(artifacts[0])
        assert row["fading_scope"] == "sample"
        assert "assumed_stream_covariance_power" in row
        assert "realized_stream_covariance_power" in row
        with (run_dir / "batches.csv").open(newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 1
    print("Synthetic train/collect/extract integration smoke test passed")


if __name__ == "__main__":
    main_smoke()
