import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm
import os


"""

callback koji sprema model checkpointove koje je pl napravio u MLFlow
tako mozemo odmah downlodad iz MLfolwa ordredjeni run

"""


class MLflowCheckpointLogger(Callback):
    def __init__(self):
        self.model_checkpoints = []
        self.checkpoint_paths = set()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Automatically find and store all ModelCheckpoint callbacks."""
        self.model_checkpoints = [
            cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        if not self.model_checkpoints:
            print("No ModelCheckpoint callbacks found!")
        else:
            print(f"Found {len(self.model_checkpoints)} ModelCheckpoint callbacks.")

    def log_checkpoints_to_mlflow(self, trainer: Trainer) -> None:
        """Helper function to log all checkpoints to MLflow."""
        if not isinstance(trainer.logger, MLFlowLogger):
            print(
                "Trainer logger is not an instance of MLFlowLogger. Skipping logging."
            )
            return

        mlflow_logger = trainer.logger
        mlflow_run_id = mlflow_logger.run_id

        print("Logging checkpoints to MLFlow...")

        for checkpoint in self.model_checkpoints:
            checkpoint_dir = checkpoint.dirpath
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                for checkpoint_file in os.listdir(checkpoint_dir):
                    file_path = os.path.join(checkpoint_dir, checkpoint_file)
                    if (
                        os.path.isfile(file_path)
                        and os.path.splitext(file_path)[1] == ".ckpt"
                    ):
                        self.checkpoint_paths.add(file_path)
            else:
                print(
                    f"Checkpoint directory {checkpoint_dir} does not exist or is empty."
                )

        # Log each checkpoint to MLFlow as an artifact.
        for file_path in tqdm(
            self.checkpoint_paths,
            desc=f"Logging {self.checkpoint_paths}",
        ):
            mlflow_logger.experiment.log_artifact(
                run_id=mlflow_run_id,
                local_path=file_path,
                artifact_path="checkpoints",
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log checkpoints at the end of training."""
        self.log_checkpoints_to_mlflow(trainer)
        mlflow.end_run(status="FINISHED")
        print("Run ended.")

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Log checkpoints if training is interrupted by an exception (e.g., Ctrl+C)."""
        print(f"Training interrupted. Logging checkpoints before exiting...")
        self.log_checkpoints_to_mlflow(trainer)
        mlflow.end_run(status="FAILED")
        print("\nRun ended after exception.")
