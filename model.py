import torch
import torch.nn as nn

latent_size = 128
nf = 32

class Generator(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.main = nn.Sequential(
            # 1 x latent_size
            nn.ConvTranspose2d(latent_size, nf * 16, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(nf * 16),
            # 16nf * 4 * 4
            nn.ConvTranspose2d(nf * 16, nf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(nf * 8),
            # 8nf * 8 * 8
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(nf * 4),
            # 4nf * 16 * 16
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(nf * 2),
            # 2nf * 32 * 32
            nn.ConvTranspose2d(nf * 2, nf * 1, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            nn.BatchNorm2d(nf * 1),
            # nf * 64 * 64
            nn.ConvTranspose2d(nf * 1, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 * 128 * 128
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.01, True),
            # nf * 64 * 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.01, True),
            # 2nf * 32 * 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.01, True),
            # 4nf * 16 * 16
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.01, True),
            # 8nf * 8 * 8
            nn.Conv2d(nf * 8, nf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(0.01, True),
            # 16nf * 4 * 4
            nn.Conv2d(nf * 16, 1, 4, 1, 0, bias=True)
            
        )
    
    def forward(self, x):
        return self.main(x)

if __name__ == "__main__":
    g = Generator()
    out = g(torch.randn(1, latent_size, 1, 1))
    print(out.shape)

    d = Discriminator()
    out = d(torch.randn(1, 3, 128, 128))
    print(out.shape)