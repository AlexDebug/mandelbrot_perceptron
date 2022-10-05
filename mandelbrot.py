
import torch
from torch import nn
import pytorch_lightning as pl
from torch import utils
from torch import optim

from os import mkdir
from shutil import rmtree

try:
    mkdir('test_results')
except FileExistsError:
    rmtree('test_results')
    mkdir('test_results')

import numpy as np
import matplotlib.pyplot as plt

reals = np.linspace(-2, 1, num=750)
imaginaries = np.linspace(-1, 1, num=500)

complex_space = []
for re in reals:
    complex_space.append(re + 1j*imaginaries)

complex_space = np.concatenate(complex_space)

def in_mandelbrot(n_steps, c):
    z = 0
    for i in range(n_steps):
        if z.real**2 + z.imag**2 > 4:
            return 0, i % n_steps
        z = z**2 + c

    return 1,i % n_steps

mandelbrot = []
for c in complex_space:
    mandelbrot.append(in_mandelbrot(500, c))

mandelbrot = np.array(mandelbrot)

plt.figure(figsize=(20,20), dpi=150)
plt.scatter(complex_space.real, complex_space.imag, c=mandelbrot[:, 1].reshape(1, -1))
plt.savefig('mandelbrot.png')

class MandelBrotDataset(utils.data.DataLoader):
    def __init__(self, complex_space, mandelbrot) -> None:
        self.complex_space = complex_space
        self.mandelbrot = mandelbrot
        self.max = mandelbrot[:, 1].max()

    def __len__(self):
        return self.complex_space.shape[0]

    def __getitem__(self, i):
        return torch.tensor([complex_space[i].real/2, complex_space[i].imag/2]).float(), torch.tensor(mandelbrot[i, 1]/self.max).float()

dataset = MandelBrotDataset(complex_space, mandelbrot)
dataloader = utils.data.DataLoader(dataset, shuffle=True, batch_size=64)

class MLP(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(2, 10),
            nn.GELU(),
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 20),
            nn.GELU(),
            nn.Linear(20, 20),
            nn.GELU(),
            nn.Linear(20, 20),
            nn.GELU(),
            nn.Linear(20, 20),
            nn.GELU(),
            nn.Linear(20, 10),
            nn.GELU(),
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 10),
            nn.GELU(),
            nn.Linear(10, 1)
        )
        self.f_loss = nn.MSELoss()
        self.count = 0

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        if batch_idx%500 == 0 or batch_idx == 0:
            d = self.forward(dataset[:][0].permute(1, 0)).detach().numpy()
            np.save(f'test_results/{self.count}.npy', d*dataset.max)
            self.count += 1
        z = self.linears(x).view(-1)
        loss = self.f_loss(z, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def forward(self, x):
        return self.linears(x)


mlp = MLP() 
trainer = pl.Trainer(max_epochs=10)
trainer.fit(mlp, dataloader)

import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(20,20), dpi=150)
imgs = []
for i in range(mlp.count):
    d = np.load(f"test_results/{i}.npy")
    im = ax.scatter(complex_space.real, complex_space.imag, c=d*dataset.max)
    imgs.append([im])

ani = animation.ArtistAnimation(fig, imgs, interval=5, blit=True,
                                repeat_delay=100)

plt.show()
ani.save('train.gif', writer=animation.PillowWriter())




