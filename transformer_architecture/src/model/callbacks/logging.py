from pytorch_lightning.callbacks import Callback

class LogFirstBatchCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._logged_train = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self._logged_val = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and not self._logged_train:
            self._log_sample(trainer, pl_module, batch, outputs, stage="train")
            self._logged_train = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and not self._logged_val:
            self._log_sample(trainer, pl_module, batch, outputs, stage="val")
            self._logged_val = True

    def _log_sample(self, trainer, pl_module, batch, outputs, stage):
        inputs, labels = batch  # adjust unpacking to your batch structure
        epoch = trainer.current_epoch

        print(f"[{stage}] epoch={epoch}")
        print(f"  input[0]:  {inputs[0]}")
        print(f"  label[0]:  {labels[0]}")
        # outputs is whatever your training_step returns
        print(f"  output[0]: {outputs}")