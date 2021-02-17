from wgan_modules.wgan_gp import WGAN_GP
from modules.datamodule import LofiDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

seed_everything(42)
dm = LofiDataModule('E:/datasets/youtube/wavfiles')
model = WGAN_GP()
trainer = Trainer(gpus=-1, auto_select_gpus=True, precision=16,
                  deterministic=True, val_check_interval=0.5,
                  automatic_optimization=False, overfit_batches=0.1,
                  callbacks=[EarlyStopping(monitor='crit_val_loss')])
trainer.fit(model, dm)