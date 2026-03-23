import pytorch_lightning as pl
import os


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N epochs, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_epoch_frequency,
        prefix="N-Epoch-Checkpoint",
    ):
        """
        Args:
            save_epoch_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
        """
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        if epoch % self.save_epoch_frequency == 0:
            filename = f"{self.prefix}_epoch={epoch}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            trainer.save_checkpoint(ckpt_path)
