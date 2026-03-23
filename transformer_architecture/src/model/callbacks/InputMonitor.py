from src.data.utils.data_utils import numpy_to_mp3

import pytorch_lightning as pl
import tempfile
import random
import mlflow
import os


class InputMonitor(pl.Callback):
    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Fetch dataset from dataloader
        dataloader = trainer.train_dataloader
        self.dataset = dataloader.dataset

        # Select a random sample
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]
        input, target, _ = sample

        # Temporarily save input and target to files
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, f"input_{trainer.current_epoch}.mp3")
            target_path = os.path.join(temp_dir, f"target_{trainer.current_epoch}.mp3")

            numpy_to_mp3(target_path, 44100, target)
            numpy_to_mp3(input_path, 44100, input)

            # Log files as artifacts in MLflow
            mlflow.log_artifact(temp_dir)
