import torch
import pytorch_lightning as pl

class PosEnc(pl.LightningModule):
    def __init__(self, d_model, depth=2, height=201, width=221):
        super().__init__()

    def forward(self, x):
        pass