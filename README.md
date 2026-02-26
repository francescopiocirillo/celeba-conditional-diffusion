# 🎭 CelebA Conditional Diffusion Model

=====================================================

🚀 A PyTorch implementation of a **DDPM-based conditional diffusion model** for **controllable face generation** on the CelebA dataset (64×64 resolution).

Developed for the **Generative AI (MSc in Computer Engineering)** course at the **University of Salerno**.

> Demonstrates practical experience in diffusion models, conditional generative modeling, UNet architectures, attention mechanisms, EMA stabilization and end-to-end training pipelines.

---

## 📌 Overview

This project implements a **conditional Denoising Diffusion Probabilistic Model (DDPM)** capable of generating face images conditioned on three semantic attributes:

* 👨 **Male / Female**
* 😊 **Smiling / Not Smiling**
* 👶 **Young / Not Young**

The model learns to generate realistic 64×64 face images from pure Gaussian noise, guided by attribute conditioning.

---

## 🧠 Model Architecture

The generative backbone is a **Conditional U-Net** trained to predict noise εₜ.

### 🔹 Core Components

* Sinusoidal **Time Embedding**
* Learnable **Condition Embedding**
* **FiLM modulation** inside residual blocks
* Multi-scale **Self-Attention** (16×16 and 8×8)
* Linear beta schedule (1000 diffusion steps)
* **EMA (Exponential Moving Average)** stabilization

### 🔄 Diffusion Process

Forward Process:

```
x₀ → xₜ (progressive noise injection)
```

Reverse Process (learned):

```
xₜ → xₜ₋₁ → ... → x₀
```

The model predicts noise at each timestep and reconstructs the clean image via iterative denoising.

---

## 🏗 Architecture Details

**Input Resolution:** 64×64  
**Base Channels:** 128  
**Timesteps:** 1000  
**Optimizer:** AdamW  
**Loss:** MSE (noise prediction objective)

The U-Net includes:

* Downsampling: 64 → 32 → 16 → 8
* Bottleneck at 8×8
* Symmetric decoder with skip connections
* Attention layers at 16×16 and 8×8 resolutions
* FiLM-based conditioning (time + attributes)

---

## 📊 Conditioning Strategy

Each sample is conditioned on a 3-dimensional binary vector:

```
[Male, Smiling, Young]
```

Example:

```
[1, 1, 0] → Male, Smiling, Not Young
```

All 8 possible attribute combinations are supported during sampling.

---

## 📂 Repository Structure

```
📦 celeba-conditional-diffusion  
├── architecture.py        # Conditional U-Net with Attention + FiLM  
├── training_lite.py        # DDPM scheduler + training loop + EMA  
├── inference.py           # Conditional sampling script  
│  
├── weights/            # Saved model weights  
│  
└── README.md
```

---

## 🧪 Training Pipeline

### 1️⃣ Dataset

Dataset used: **CelebA**

Attributes extracted:

* #20 → Male
* #31 → Smiling
* #39 → Young

Images are:

* Resized to 64×64
* Center cropped
* Normalized to [-1, 1]

---

### 2️⃣ Training

Run:

```
python training_lite.py
```

Features:

* Random timestep sampling
* Forward diffusion noise injection
* Noise prediction objective
* Gradient clipping
* EMA model tracking
* Periodic sample generation
* Automatic checkpoint saving

Checkpoints saved in:

```
weights/latest.pt
```

---

### 3️⃣ Inference

Edit the conditioning vector inside:

```
inference.py
```

Then run:

```
python inference.py
```

The script:

* Loads the trained checkpoint
* Restores the scheduler
* Generates N samples with the same conditioning
* Saves a grid image in `/generated`

---

## 🔬 Technologies Used

* PyTorch
* Torchvision
* CelebA Dataset
* UNet Architecture
* Self-Attention
* DDPM Scheduler (custom implementation)
* EMA (Stabilized Training)
* AdamW Optimizer

---

## 🎓 Academic Context

Developed as a **Project Work** for:

**Generative AI – MSc in Computer Engineering**  
University of Salerno  
Academic Year 2025/2026

The original assignment required multiple generative approaches; this repository contains the **Diffusion Model implementation**.

The other 2 implementations can be found [here]().

---

## 💡 Key Challenges Addressed

* Stable diffusion training from scratch
* Conditioning injection inside UNet blocks
* Attention integration at low resolutions
* Reverse diffusion numerical stability
* EMA-based sampling stabilization
* Memory-efficient training at 64×64

---

## ⭐ Final Note

This project highlights:

* Deep understanding of diffusion models
* Conditional generative modeling
* Architectural design of UNet with attention
* Training stabilization techniques
* Full generative pipeline implementation from scratch

If you find it interesting, feel free to ⭐ the repository.

---

## 📈 SEO Tags

```
Diffusion Model, DDPM, Conditional Diffusion, CelebA Diffusion, Face Generation AI, Conditional UNet, Generative AI MSc Project, PyTorch Diffusion Implementation, Noise Prediction Model, Attribute Conditioned Generation, Denoising Diffusion Probabilistic Model, Self Attention UNet, EMA Diffusion Training, Computer Vision Generative Models
```

---

## 📄 License

This project is licensed under the **MIT License**.

> Use it, build on it, experiment with it, just don’t blame the diffusion process if it generates something unexpected 😄
