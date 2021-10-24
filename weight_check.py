import os
import re
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import Generator, Discriminator, latent_size

device = "cuda" if torch.cuda.is_available() else "cpu"
pat = re.compile("e_([0-9]+).ckpt")
weights = sorted([w for w in os.listdir("weights") if w.startswith("e_") and w.endswith(".ckpt")], key=lambda w: int(pat.match(w).group(1)))

netG = Generator().to(device)
netD = Discriminator().to(device)

def load_weight(name: str):
    ckpt = torch.load(os.path.join("weights", f"{name}.ckpt"))
    netG.load_state_dict(ckpt["net_g"])
    netD.load_state_dict(ckpt["net_d"])

noise = torch.randn(64, latent_size, 1, 1).to(device)

for weight in weights:
    print(weight)
    load_weight(weight[:-5])
    netG.eval() 
    outs = netG(noise)
    output = make_grid((outs + 1)/2.0)
    plt.imshow(output.cpu().permute(1, 2, 0))
    plt.show()



