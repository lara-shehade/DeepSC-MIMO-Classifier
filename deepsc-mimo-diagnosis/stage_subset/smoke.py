"""Small CPU-only gate for the post-hoc subset analysis."""

import json
from pathlib import Path
import tempfile

import torch

from analysis.neural_diagnostics import CLASSES, StageClassifier, scan_records
from stage_subset.ablation import (
    MULTI_SUBSETS,
    PILOT_INDEX,
    SEEDS,
    STAGES,
    SUBSETS,
    TASKS,
    TRAINING,
    SubsetTCN,
    evaluate_fold,
    subset_name,
    summarize,
    write_json_once,
)


def make_traces(root):
    generator = torch.Generator().manual_seed(761)
    for seed in SEEDS:
        for label_index, label in enumerate(CLASSES):
            directory = root / label / "seed{}".format(seed) / "artifacts"
            directory.mkdir(parents=True)
            real = torch.randn(1, 16, 2, generator=generator)
            imag = torch.randn(1, 16, 2, generator=generator)
            torch.save({
                "diagnosis_label": label,
                "holdout": False,
                "physical_seed": seed,
                "sample_ids": torch.tensor([seed * 100 + label_index]),
                "source_lengths": torch.tensor([4]),
                "R3_generated_lengths": torch.tensor([5]),
                "R0_noisy_received": torch.complex(real, imag),
                "R1_equalized_decoder_input": torch.randn(1, 4, 16, generator=generator),
                "R2_channel_decoder_output": torch.randn(1, 4, 128, generator=generator),
                "R3_free_running_decoder_hidden": torch.randn(1, 5, 128, generator=generator),
            }, directory / "batch_000.pt")


def smoke():
    assert len(SUBSETS) == 15 and len(MULTI_SUBSETS) == 11 and len(TASKS) == 55
    assert TASKS[PILOT_INDEX] == (STAGES, SEEDS[0])
    reference = StageClassifier("tcn", "R3").eval()
    singleton = SubsetTCN(("R3",)).eval()
    singleton.encoders["R3"].load_state_dict(reference.encoder.state_dict())
    singleton.classifier.load_state_dict(reference.classifier.state_dict())
    values = {"R3": torch.randn(2, 5, 128)}
    masks = {"R3": torch.tensor([[True] * 5, [True] * 3 + [False] * 2])}
    with torch.no_grad():
        assert torch.allclose(reference(values, masks), singleton(values, masks))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        trace_root = root / "development"
        make_traces(trace_root)
        records = scan_records(trace_root)
        assert len(records) == 35
        pilot = evaluate_fold(records, STAGES, SEEDS[0], torch.device("cpu"), epochs=1)
        assert pilot["train_samples"] == 28 and pilot["test_samples"] == 7
        assert len(pilot["confusion_matrix"]) == len(CLASSES)
        assert all(len(row) == len(CLASSES) for row in pilot["confusion_matrix"])

        output_root = root / "subset_analysis"
        singleton_root = root / "existing_neural_folds"
        for stages in SUBSETS:
            name = subset_name(stages)
            for seed in SEEDS:
                if len(stages) == 1:
                    write_json_once(
                        singleton_root / "tcn_{}_seed{}.json".format(stages[0], seed),
                        {**pilot, "architecture": "tcn", "stage": stages[0],
                         "held_out_physical_seed": seed},
                    )
                else:
                    write_json_once(
                        output_root / "folds" / "{}_seed{}.json".format(name, seed),
                        {**pilot, "subset": name, "stages": list(stages),
                         "held_out_physical_seed": seed,
                         "training": TRAINING,
                         "parameter_count": sum(
                             parameter.numel() for parameter in
                             SubsetTCN(stages).parameters()
                         )},
                    )
        summarize(output_root, singleton_root)
        report = json.loads(
            (output_root / "stage_subset_report.json").read_text(encoding="utf-8")
        )
        assert len(report["folds"]) == 75 and len(report["aggregate"]) == 15
        assert report["holdout_used"] is False
    print("PASS: subset mapping, singleton parity, training, and aggregation")


if __name__ == "__main__":
    smoke()
