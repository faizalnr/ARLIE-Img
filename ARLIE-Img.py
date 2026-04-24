#**********************************************************************
# code executed by Faizal Nujumudeen
# Presidency University, Bengaluru

import numpy as np
import cv2
import hashlib
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log2

# =========================
# Metrics
# =========================

def entropy(img):
    hist = np.histogram(img.flatten(), bins=256, range=[0,256])[0]
    prob = hist / np.sum(hist)
    return -np.sum([p*log2(p) for p in prob if p > 0])

def correlation_val(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = gray[:, :-1].flatten()
    y = gray[:, 1:].flatten()
    return np.corrcoef(x, y)[0,1]

def npcr_uaci(img1, img2):
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.mean(np.abs(img1.astype(int) - img2.astype(int)) / 255) * 100
    return npcr, uaci

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float))**2)
    if mse == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))

# =========================
# Key Generation
# =========================

def generate_key(image):
    ent = entropy(image)
    mean = np.mean(image)
    img_hash = hashlib.sha256(image.tobytes()).hexdigest()
    seed = hashlib.sha256((str(ent)+str(mean)+img_hash).encode()).hexdigest()
    return seed

# =========================
# Stable Chaotic System
# =========================

def chaotic_sequence(n, seed):
    x = int(seed[:8], 16) / (2**32)
    y = int(seed[8:16], 16) / (2**32)

    a, b = 3.99, 3.98
    seq = np.zeros(n)

    for i in range(n):
        x = (np.sin(np.pi * a * y) + b * x * (1 - x)) % 1
        y = (np.sin(np.pi * b * x) + a * y * (1 - y)) % 1

        if x <= 0 or x >= 1:
            x = (x % 1 + 1e-6)
        if y <= 0 or y >= 1:
            y = (y % 1 + 1e-6)

        seq[i] = x

    return seq

# =========================
# ROI Detection
# =========================

def get_roi_mask(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    grad = cv2.Laplacian(blur, cv2.CV_64F)
    T = np.mean(grad) + 0.5*np.std(grad)
    mask = (grad > T).astype(np.uint8)
    return mask

# =========================
# Encryption
# =========================

def encrypt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = get_roi_mask(gray)

    mask3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2).flatten()

    seed = generate_key(gray)

    flat = image.flatten()

    roi_indices = np.where(mask3 == 1)[0]
    roi_values = flat[roi_indices]

    chaos_roi = chaotic_sequence(len(roi_indices), seed)
    perm_roi = np.argsort(chaos_roi)

    permuted = flat.copy()
    permuted[roi_indices] = roi_values[perm_roi]

    n = len(flat)
    chaos_full = chaotic_sequence(n, seed)

    cipher = np.zeros_like(permuted)

    for i in range(n):
        prev = cipher[i-1] if i > 0 else 0
        f = ((prev << 3) & 0xFF) ^ (prev >> 2)
        cipher[i] = permuted[i] ^ int(chaos_full[i]*255) ^ f

    return cipher.reshape(image.shape), seed, mask

# =========================
# Decryption
# =========================

def decrypt(cipher_img, seed, mask):
    mask3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2).flatten()

    flat = cipher_img.flatten()
    n = len(flat)

    chaos_full = chaotic_sequence(n, seed)

    permuted = np.zeros_like(flat)

    for i in range(n):
        prev = flat[i-1] if i > 0 else 0
        f = ((prev << 3) & 0xFF) ^ (prev >> 2)
        permuted[i] = flat[i] ^ int(chaos_full[i]*255) ^ f

    roi_indices = np.where(mask3 == 1)[0]

    chaos_roi = chaotic_sequence(len(roi_indices), seed)
    perm_roi = np.argsort(chaos_roi)
    inv_perm_roi = np.argsort(perm_roi)

    original = permuted.copy()
    original[roi_indices] = permuted[roi_indices][inv_perm_roi]

    return original.reshape(cipher_img.shape)

# =========================
# Plot Functions
# =========================

def correlation_scatter(images, titles, folder):
    plt.figure(figsize=(15,5))
    for i, (img, title) in enumerate(zip(images, titles)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = gray[:, :-1].flatten()
        y = gray[:, 1:].flatten()
        plt.subplot(1,3,i+1)
        plt.scatter(x, y, s=1)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{folder}/correlation_scatter.png")
    plt.close()

def histogram_plot(images, titles, folder):
    plt.figure(figsize=(10,6))
    for img, title in zip(images, titles):
        plt.hist(img.flatten(), bins=256, alpha=0.5, label=title)
    plt.legend()
    plt.savefig(f"{folder}/histogram.png")
    plt.close()

def attack_visual(original, encrypted, folder):
    npcr, uaci = npcr_uaci(original[:,:,0], encrypted[:,:,0])

    plt.figure()
    plt.bar(["NPCR","UACI"], [npcr, uaci])
    plt.title("Attack Metrics")
    plt.savefig(f"{folder}/attack_metrics.png")
    plt.close()

# =========================
# Main
# =========================

def save_results(input_path):
    img = cv2.imread(input_path)
    name = os.path.splitext(os.path.basename(input_path))[0]

    folder = f"output_{name}"
    os.makedirs(folder, exist_ok=True)

    enc, seed, mask = encrypt(img)
    dec = decrypt(enc, seed, mask)

    cv2.imwrite(f"{folder}/original.png", img)
    cv2.imwrite(f"{folder}/encrypted.png", enc)
    cv2.imwrite(f"{folder}/decrypted.png", dec)

    histogram_plot([img, enc, dec],
                   ["Original","Encrypted","Decrypted"], folder)

    correlation_scatter([img, enc, dec],
                        ["Original","Encrypted","Decrypted"], folder)

    attack_visual(img, enc, folder)

    print("Saved results in:", folder)

# =========================
# Run
# =========================

if __name__ == "__main__":
    save_results("Brain.jpg")

# "If you want to shine like a sun, first burn like a sun" - Dr. APJ Abdul Kalam.
# Success is a continuous process
#**********************************************************************