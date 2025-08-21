import os, requests

# Create folder
save_path = "kodak_dataset"
os.makedirs(save_path, exist_ok=True)

# Download 24 Kodak images
base_url = "http://r0k.us/graphics/kodak/kodak"
for i in range(1, 25):  # 24 images (01 → 24)
    url = f"{base_url}/kodim{i:02d}.png"
    file_path = os.path.join(save_path, f"kodim{i:02d}.png")
    if not os.path.exists(file_path):  # avoid redownload
        r = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {file_path}")

print("✅ All Kodak images downloaded into", save_path)
