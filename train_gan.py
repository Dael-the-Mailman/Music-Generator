import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile

from tqdm import tqdm
from wgan_modules.critic import Critic
from wgan_modules.generator import Generator
from modules.datamodule import TrainSpecLoader, TestSpecLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
SPEC_SIZE = 64
BATCH_SIZE = 64
Z_DIM = 100
AUDIO_CHANNELS = 1
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
PROPORTION = 0.25

dataset = TestSpecLoader('E:/datasets/youtube/wavfiles', SPEC_SIZE)
loader = DataLoader(dataset, shuffle=False)

gen = Generator(Z_DIM, AUDIO_CHANNELS, FEATURES_GEN).to(device)
critic = Critic(AUDIO_CHANNELS, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen  = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
# scaler = amp.GradScaler()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/AudioGAN/real")
# writer_fake = SummaryWriter(f"logs/AudioGAN/fake")
step = 0

gen.train()
critic.train()

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train(critic, opt_critic, gen, opt_gen, real, cur_batch_size, Z_DIM, device):
    # Train Critic: max E[critic(real)] - E[critic(fake)]
    # equivalent to minimizing the negative of that
    for _ in range(CRITIC_ITERATIONS):
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        critic_real = critic(real).reshape(-1)
        critic_fake = critic(fake).reshape(-1)
        gp = gradient_penalty(critic, real, fake, device=device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake) + LAMBDA_GP * gp)
        )
        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()
    
    # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
    gen_fake = critic(fake).reshape(-1)
    loss_gen = -torch.mean(gen_fake)
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    return loss_critic, loss_gen

for epoch in range(NUM_EPOCHS):
    history = pd.DataFrame(columns=["index", "loss"])
    for batch_idx, (real, _) in tqdm(enumerate(loader)):
        real = torch.from_numpy(np.stack(real)).to(device)
        cur_batch_size = real.shape[0]

        if cur_batch_size > 200:
            print('Song too large')
            continue

        loss_critic, loss_gen = train(critic, opt_critic, gen, opt_gen, real, cur_batch_size, Z_DIM, device)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )

        history = history.append({"index": batch_idx, "loss": loss_critic.item()}, ignore_index=True)

    worst_df = history[history["loss"] <= history["loss"].quantile(PROPORTION)]

    print("\nNow training on worst songs\n")
    
    worst_idx = 1
    for song_idx, row in worst_df.iterrows():
        idx = int(song_idx)
        real = torch.from_numpy(np.stack(dataset[idx][0])).unsqueeze(1).to(device)
        cur_batch_size = real.shape[0]
        
        # For redundancy
        if cur_batch_size > 200:
            print('Song too large')
            continue

        loss_critic, loss_gen = train(critic, opt_critic, gen, opt_gen, real, cur_batch_size, Z_DIM, device)
        
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Worst Song {worst_idx}/{len(worst_df)} \
              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )
        worst_idx += 1
        # Print losses occasionally and print to tensorboard
        # if batch_idx % 100 == 0 and batch_idx > 0:
        #     print(
        #         f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
        #           Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        #     )

    with torch.no_grad():
        fake = gen(fixed_noise)
        # take out (up to) 32 examples
        img_grid_real = torchvision.utils.make_grid(real[:32].cpu()) #, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:32].cpu()) #, normalize=True)

        plt.imshow(img_grid_real.log().permute(1,2,0).numpy(), cmap='gist_heat')
        plt.savefig(f"logs/Epoch {epoch} Real.png")
        plt.imshow(img_grid_fake.log().permute(1,2,0).numpy(), cmap='gist_heat')
        plt.savefig(f"logs/Epoch {epoch} Fake.png")

        y_prime = []
        for chunk in tqdm(fake[:32].cpu().numpy()):
            out = librosa.feature.inverse.mel_to_audio(chunk.squeeze(0))
            y_prime.append(out)
        soundfile.write(f'logs/Epoch {epoch}.wav', np.concatenate(y_prime), 22050)
        # out = librosa.feature.inverse.mel_to_audio(chunk, sr=sr)
        # y_prime.append(out)
        # torchvision.utils.save_image(img_grid_real, f"Epoch {epoch} Real.png")
        # torchvision.utils.save_image(img_grid_fake, f"Epoch {epoch} Fake.png")
                # writer_real.add_image("Real", img_grid_real, global_step=step)
                # writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        #     step += 1
