import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from channels import MIMOChannel, MIMOConfig
from experiment_data import ExperimentDataWriter, SCHEMA_VERSION
from main import rng_state_dict, restore_rng_state
from models.transceiver import DeepSC
from utils import forward_pipeline, initNetParams


class PipelineTests(unittest.TestCase):
    def test_rng_resume_moves_saved_states_to_cpu(self):
        class SavedDeviceState:
            def __init__(self, value):
                self.value = value
                self.cpu_calls = 0

            def cpu(self):
                self.cpu_calls += 1
                return self.value

        state = rng_state_dict()
        cpu_state = SavedDeviceState(state["torch_cpu"])
        cuda_state = SavedDeviceState(state["torch_cpu"].clone())
        state["torch_cpu"] = cpu_state
        state["torch_cuda"] = [cuda_state]
        with mock.patch("main.torch.cuda.is_available", return_value=True), \
                mock.patch("main.torch.cuda.device_count", return_value=1), \
                mock.patch("main.torch.cuda.set_rng_state_all") as set_cuda:
            restore_rng_state(state)
        self.assertEqual(cpu_state.cpu_calls, 1)
        self.assertEqual(cuda_state.cpu_calls, 1)
        set_cuda.assert_called_once()
        self.assertIs(set_cuda.call_args.args[0][0], cuda_state.value)

    def _model(self):
        model = DeepSC(1, 20, 20, 20, 20, 8, 2, 16, 0.0)
        return initNetParams(model)

    def test_forward_trace_and_writer(self):
        torch.manual_seed(4)
        model = self._model()
        source = torch.tensor([[1, 4, 5, 2, 0], [1, 7, 2, 0, 0]])
        criterion = nn.CrossEntropyLoss(reduction="none")
        channel = MIMOChannel(MIMOConfig(fading_scope="sample"), seed=11)
        loss, trace = forward_pipeline(
            model, source, source, 0.1, 0, criterion,
            channel_model=channel, collect_trace=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(trace.tx_normalized.shape), (2, 5, 16))
        self.assertEqual(tuple(trace.channel.tx_grouped.shape), (2, 20, 2))
        self.assertEqual(tuple(trace.channel.true_h.shape), (2, 1, 2, 2))

        with tempfile.TemporaryDirectory() as temporary:
            writer = ExperimentDataWriter(
                temporary, "baseline", run_id="unit", metadata={"purpose": "test"}
            )
            artifact = writer.save_batch(
                trace, "test", 0, 0, 0, sample_ids=torch.tensor([10, 11]),
                extra_metrics={"snr_db": 10.0},
            )
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertEqual(tuple(payload["channel"]["true_h"].shape), (2, 1, 2, 2))
            metadata = json.loads((writer.run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], SCHEMA_VERSION)
            with (writer.run_dir / "batches.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["snr_db"], "10.0")
            self.assertEqual(len((writer.run_dir / "manifest.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
