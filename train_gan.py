from wgan_modules.wgan_gp import WGAN_GP
from modules.datamodule import LofiDataModule
from pytorch_lightning.trainer.trainer import Trainer

dm = LofiDataModule()
model = WGAN_GP()
trainer = Trainer(gpus=1, precision=16)
print("Success 😎")