import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    gradient_penalty,
    save_checkpoint,
    load_checkpoint,
    generate_examples
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
from lofidataset import LofiDataset

torch.backends.cudnn.benchmark = True


def get_loader(mel_size):
    batch_size = config.BATCH_SIZES[int(log2(mel_size / 4))]
    dataset = LofiDataset(config.PATH, n_mels=mel_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    scaler_gen,
    scaler_critic
):
    loop = tqdm(loader, leave=True)
    for batch_idx, real in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        print(cur_batch_size)

        import sys
        sys.exit()

        noise = torch.randn(cur_batch_size, config.Z_DIM,
                            1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha,
                                  step, device=config.DEVICE)
            loss_critic = (
                - (torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / \
            (len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.5)
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item()
        )

    return alpha


def main():
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()

    step = int(log2(config.START_TRAIN_AT_IMG_SIZE/4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        print(f"Image size: {4*2**step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}")
            alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                scaler_gen,
                scaler_critic
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic,
                                filename=config.CHECKPOINT_CRITIC)

        step += 1


if __name__ == '__main__':
    main()
