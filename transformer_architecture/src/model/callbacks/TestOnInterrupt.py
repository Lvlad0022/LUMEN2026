import signal
import pytorch_lightning as pl

class TestOnInterrupt(pl.Callback):
    def __init__(self):
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self._handle_signal)  # Listen for SIGUSR1
        
    def _handle_signal(self, signum, frame):
        print(f"Signal {signum} received! Triggering testing event.")
        self.signal_received = True

    def on_train_batch_end(self, trainer:pl.Trainer, pl_module:pl.LightningModule, outputs, batch, batch_idx):
        if self.signal_received:
            print("External signal event triggered. Running testing.")
            trainer.test(
                model=pl_module,
                dataloaders=trainer.datamodule.test_dataloader()
                )
            raise KeyboardInterrupt

    def on_validation_batch_end(self, trainer:pl.Trainer, pl_module:pl.LightningModule, outputs, batch, batch_idx):
        if self.signal_received:
            print("External signal event triggered. Running testing.")
            trainer.test(
                model=pl_module,
                dataloaders=trainer.datamodule.test_dataloader()
                )
            raise KeyboardInterrupt

    