import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp

from tqdm import tqdm
from wgan_modules.critic import Critic
from wgan_modules.generator import Generator
from modules.datamodule import TrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
Z_DIM = 100
AUDIO_CHANNELS = 2
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

loader = DataLoader(TrainDataset('E:/datasets/youtube/wavfiles'), shuffle=True)

gen = Generator(Z_DIM, AUDIO_CHANNELS, FEATURES_GEN).to(device)
crit = Critic(AUDIO_CHANNELS, FEATURES_CRITIC).to(device)

opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_c = optim.Adam(crit.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
scaler = amp.GradScaler()

fixed_noise = torch.randn(64, Z_DIM, 1).to(device)
# writer_real = SummaryWriter(f"logs/AudioGAN/real")
# writer_fake = SummaryWriter(f"logs/AudioGAN/fake")
step = 0

gen.train()
crit.train()

def gradient_penalty(critic, real, fake, device="cpu"):
    with amp.autocast():
        BATCH_SIZE, C, S = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, S).to(device)
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=scaler.scale(mixed_scores),
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    with amp.autocast():
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

for epoch in range(NUM_EPOCHS):
    for batch_idx, (song, _) in tqdm(enumerate(loader)):
        song = song.squeeze(0).to(device)
        song = torch.split(song, BATCH_SIZE) # Splits the song to decrease memory usage
        
        for real in song:
            curr_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                with amp.autocast():
                    noise = torch.randn(curr_batch_size, Z_DIM, 1).to(device)
                    fake = gen(noise)
                    crit_real = crit(real).reshape(-1)
                    crit_fake = crit(fake).reshape(-1)
                    gp = gradient_penalty(crit, real, fake, device=device)
                    loss_crit = -(
                        torch.mean(crit_real) - torch.mean(crit_fake) + LAMBDA_GP * gp
                    )
                crit.zero_grad()
                scaler.scale(loss_crit).backward(retain_graph=True)
                scaler.step(opt_c)
                scaler.update()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with amp.autocast():
                gen_fake = crit(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            scaler.scale(loss_gen).backward()
            scaler.step(opt_g)
            scaler.update()

        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_crit:.4f}, loss G: {loss_gen:.4f}"
            )
    break
