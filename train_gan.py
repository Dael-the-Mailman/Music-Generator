from wgan_modules.wgan_gp import WGAN_GP
from modules.datamodule import LofiDataModule
from pytorch_lightning import Trainer, seed_everything

seed_everything(42)
dm = LofiDataModule('E:/datasets/youtube/wavfiles')
model = WGAN_GP()
trainer = Trainer(gpus=-1, auto_select_gpus=True, precision=16,
                  deterministic=True, check_val_every_n_epoch=10,
                  automatic_optimization=False, overfit_batches=0.1)
trainer.fit(model, dm)