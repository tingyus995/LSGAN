import os
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import AnimeDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Generator, Discriminator, latent_size
from torch.backends import cudnn
cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = AnimeDataset(r"C:\Users\TingYu\projects\anime_faces", transform=T.Compose([
    T.ToTensor(),
    T.Resize((128, 128)),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]))

loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)

netG = Generator().to(device)
netD = Discriminator().to(device)

scalarG = GradScaler()
scalarD = GradScaler()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

def save_weight(name: str):
    ckpt = {
        "net_g": netG.state_dict(),
        "net_d": netD.state_dict(),
        "scalar_g": scalarG.state_dict(),
        "scalar_d": scalarD.state_dict(),
    }

    torch.save(ckpt, os.path.join("weights", f"{name}.ckpt"))

def load_weight(name: str):

    ckpt = torch.load(os.path.join("weights", f"{name}.ckpt"))
    netG.load_state_dict(ckpt["net_g"])
    netD.load_state_dict(ckpt["net_d"])
    scalarG.load_state_dict(ckpt["scalar_g"])
    scalarD.load_state_dict(ckpt["scalar_d"])

    print("weight loaded.")

if __name__ == "__main__":
    # training loop
    # load_weight("e_40")
    num_epochs = 150
    criterion = nn.MSELoss()
    optimD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    writer = SummaryWriter()

    n_critics = 2
    n_iters = 0

    try:
        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")
            for index, imgs in enumerate(tqdm(loader)):
                imgs_gpu = imgs.to(device) 
                b_size = imgs.size(0)
                # train discriminator

                if index % n_critics == 0:
                    netD.zero_grad()

                    with autocast():
                        # fake
                        noise = torch.randn(b_size, latent_size, 1, 1).to(device)
                        generated_imgs = netG(noise)
                        outs = netD(generated_imgs.detach())
                        loss_D_fake = criterion(outs, torch.zeros_like(outs))
                        # real
                        outs = netD(imgs_gpu)
                        loss_D_real = criterion(outs, torch.ones_like(outs))

                        loss_D = loss_D_fake + loss_D_real

                    scalarD.scale(loss_D).backward()
                    scalarD.step(optimD)
                    scalarD.update()
                    writer.add_scalar("lossD", loss_D.item(), n_iters)

                # train generator
                netG.zero_grad()
                noise = torch.randn(b_size, latent_size, 1, 1).to(device)

                with autocast():
                    generated_imgs = netG(noise)
                    outs = netD(generated_imgs)
                    loss_G = criterion(outs, torch.ones_like(outs))

                scalarG.scale(loss_G).backward()
                scalarG.step(optimG)
                scalarG.update()
                if n_iters % 200 == 0:
                    writer.add_image("G(z)", make_grid((generated_imgs+1) / 2.0), n_iters)
                writer.add_scalar("lossG", loss_G.item(), n_iters)
                n_iters += 1
            if epoch % 5 == 0:
                save_weight(f"e_{epoch}")
    except KeyboardInterrupt:
        save_weight("interrupt")
    
    save_weight("final")