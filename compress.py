import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# PSNR function
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compress_image(input_image_path, output_compressed_path='compressed/latent.pt', output_recon_path='compressed/reconstructed.png'):
    latent_dim = 128  # Must match training

    # Load models
    from models.autoencoder import Encoder, Decoder
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    encoder.load_state_dict(torch.load('checkpoints/encoder.pth'))
    decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
    encoder.eval()
    decoder.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Transform input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(input_image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # Add batch dim
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Compress (encode)
    with torch.no_grad():
        latent = encoder(img_tensor)
        recon = decoder(latent)

    # Save compressed latent
    os.makedirs('compressed', exist_ok=True)
    torch.save(latent.cpu(), output_compressed_path)
    print(f'Compressed latent saved to {output_compressed_path}')

    # Save reconstructed image
    recon_img = transforms.ToPILImage()(recon.squeeze(0).cpu())
    recon_img.save(output_recon_path)
    print(f'Reconstructed image saved to {output_recon_path}')

    # Compute metrics
    original_np = np.array(image.resize((256, 256))) / 255.0  # Normalize
    recon_np = np.array(recon_img) / 255.0
    psnr_val = psnr(original_np, recon_np)
    ssim_val = ssim(original_np, recon_np, multichannel=True, channel_axis=-1, data_range=1.0)
    print(f'PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')

if __name__ == '__main__':
    # Example usage: replace with your image path
    input_path = 'Data/Kodak/kodim01.png'  # Change this
    compress_image(input_path)