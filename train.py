import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt 


class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# PSNR function
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Training function
def train():
    epochs = 100
    batch_size = 4  
    learning_rate = 0.001
    latent_dim = 128 

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = KodakDataset(root_dir='Data/Kodak', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    from models.autoencoder import Encoder, Decoder
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(epochs):
        for data in dataloader:
            img = data
            if torch.cuda.is_available():
                img = img.cuda()

            latent = encoder(img)
            output = decoder(latent)

            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            original = img[0].cpu().numpy().transpose(1,2,0)
            recon = output[0].cpu().numpy().transpose(1,2,0)
            psnr_val = psnr(original, recon)  
            ssim_val = ssim(original, recon, multichannel=True, channel_axis=-1, data_range=1.0)
            print(f'PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')

    torch.save(encoder.state_dict(), 'checkpoints/encoder.pth')
    torch.save(decoder.state_dict(), 'checkpoints/decoder.pth')
    print('Training complete. Models saved.')

if __name__ == '__main__':
    train()