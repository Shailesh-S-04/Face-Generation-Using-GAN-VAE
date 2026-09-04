# Face Generation Using GAN & VAE

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit 1.30+](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset: CelebA](https://img.shields.io/badge/Dataset-CelebA-success.svg)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

An interactive, academic-grade project demonstration and comparative analysis between **Variational Autoencoders (VAE)** and **Basic Generative Adversarial Networks (GAN)** for synthetic human face generation on the **CelebA** dataset.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [Project Architecture & Directory Structure](#project-architecture--directory-structure)
- [Model Architectures & Specifications](#model-architectures--specifications)
  - [1. Variational Autoencoder (VAE)](#1-variational-autoencoder-vae)
  - [2. Basic Convolutional GAN](#2-basic-convolutional-gan)
- [Comprehensive Training & Evaluation Metrics](#comprehensive-training--evaluation-metrics)
  - [Basic GAN Quantitative Results](#basic-gan-quantitative-results)
  - [VAE Quantitative Training Progression](#vae-quantitative-training-progression)
- [Model Comparison Matrix](#model-comparison-matrix)
- [Qualitative Analysis & Observations](#qualitative-analysis--observations)
- [Installation & Execution](#installation--execution)
- [Streamlit Interactive Web App Structure](#streamlit-interactive-web-app-structure)

---

## Executive Overview

| Attribute | Variational Autoencoder (VAE) | Basic Convolutional GAN |
| :--- | :--- | :--- |
| **Primary Goal** | Probabilistic latent manifold learning & reconstruction | Adversarial game-theoretic image synthesis |
| **Native Resolution** | $64 \times 64 \times 3$ *(presented at $128 \times 128$ via Bicubic upscaling)* | $128 \times 128 \times 3$ *(Native output)* |
| **Latent Space** | $64$-Dimensional continuous Gaussian ($\mu, \sigma^2$) | $128$-Dimensional Gaussian noise ($z \sim \mathcal{N}(0, I)$) |
| **Total Parameters** | **$2,172,099$** ($\sim 2.17\text{M}$) | **$15,016,548$** ($\sim 15.02\text{M}$) |
| **Training Objective** | Reconstruction Loss (MSE / Feature Matching) + KL Divergence | Minimax Adversarial Binary Cross-Entropy |
| **Hardware** | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) / CPU fallback | CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) / CPU fallback |

---

## Project Architecture & Directory Structure

```
D:\Face-Generation-Using-GAN-VAE
│
├── app.py                             # Upgraded Streamlit presentation application
├── README.md                          # Comprehensive project documentation & metrics
├── requirements.txt                   # Environment dependencies
│
├── .streamlit/
│   └── config.toml                    # Server & file watcher configuration
│
├── models/                            # Trained VAE model weights & latent distribution
│   ├── encoder_final.pth              # VAE Encoder weights (1,215,520 params | 4.88 MB)
│   ├── decoder_final.pth              # VAE Decoder weights (956,579 params | 3.84 MB)
│   ├── discriminator_final.pth        # VAE-side training Discriminator (11.08 MB)
│   └── latent_distribution.pth        # Learned empirical latent distribution (mean & std)
│
└── latest_results/                    # Trained Basic GAN weights & empirical evaluations
    ├── generator_final.pth            # Basic GAN Generator (8,055,587 params | 32.31 MB)
    ├── discriminator_final.pth        # Basic GAN Discriminator (6,960,961 params | 27.87 MB)
    ├── gan_training_history.csv       # Loss log per epoch across 30 epochs
    ├── gan_training_history.pth       # Serialized loss history dictionary
    ├── gan_evaluation_metrics.csv     # Final evaluation metrics & accuracy breakdown
    ├── GAN_EVALUATION_REPORT.txt      # Text summary of training & discriminator evaluation
    ├── gan_training_losses.png        # Recorded training curves (Generator vs Discriminator)
    └── generated_evaluation_grid.png  # High-resolution 128x128 evaluation sample grid
```

> **Important Note on Checkpoint Segregation:**
> `models/discriminator_final.pth` ($11.08\text{ MB}$) is the auxiliary discriminator used during VAE-GAN training. `latest_results/discriminator_final.pth` ($27.87\text{ MB}$) is the dedicated $128 \times 128$ Basic GAN discriminator. These are distinct architectures and are strictly segregated.

---

## Model Architectures & Specifications

### 1. Variational Autoencoder (VAE)

- **Input Dimension:** $3 \times 64 \times 64$
- **Latent Dimension:** $64$
- **Encoder Architecture:**
  - $\text{Conv2d}(3 \to 32, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU}(0.2)$
  - $\text{Conv2d}(32 \to 64, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU}(0.2)$
  - $\text{Conv2d}(64 \to 128, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU}(0.2)$
  - $\text{Conv2d}(128 \to 256, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU}(0.2)$
  - Output Feature Map: $256 \times 4 \times 4 = 4096$
  - Fully Connected Heads: $\mu \in \mathbb{R}^{64}$ and $\log\sigma^2 \in \mathbb{R}^{64}$
- **Decoder Architecture:**
  - Fully Connected: $\text{Linear}(64 \to 4096) \to \text{Reshape}(256 \times 4 \times 4)$
  - $\text{ConvTranspose2d}(256 \to 128, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{ReLU}()$
  - $\text{ConvTranspose2d}(128 \to 64, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{ReLU}()$
  - $\text{ConvTranspose2d}(64 \to 32, k=4, s=2, p=1) \to \text{BatchNorm2d} \to \text{ReLU}()$
  - $\text{ConvTranspose2d}(32 \to 3, k=4, s=2, p=1) \to \text{Tanh}()$
- **Parameter Breakdown:**
  - **Encoder Parameters:** $1,215,520$
  - **Decoder Parameters:** $956,579$
  - **Total VAE Parameters:** **$2,172,099$** ($\sim 2.17\text{M}$)

### 2. Basic Convolutional GAN

- **Input Dimension:** Latent noise vector $z \sim \mathcal{N}(0, I_{128})$
- **Output Dimension:** $3 \times 128 \times 128$
- **Generator Architecture:**
  - Fully Connected: $\text{Linear}(128 \to 8192) \to \text{BatchNorm1d}(8192) \to \text{ReLU}() \to \text{Reshape}(512 \times 4 \times 4)$
  - $\text{ConvTranspose2d}(512 \to 512, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(512) \to \text{ReLU}() \to (8 \times 8)$
  - $\text{ConvTranspose2d}(512 \to 256, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(256) \to \text{ReLU}() \to (16 \times 16)$
  - $\text{ConvTranspose2d}(256 \to 128, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(128) \to \text{ReLU}() \to (32 \times 32)$
  - $\text{ConvTranspose2d}(128 \to 64, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(64) \to \text{ReLU}() \to (64 \times 64)$
  - $\text{ConvTranspose2d}(64 \to 32, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(32) \to \text{ReLU}() \to (128 \times 128)$
  - $\text{Conv2d}(32 \to 3, k=3, s=1, p=1) \to \text{Tanh}() \to (128 \times 128 \times 3)$
- **Discriminator Architecture:**
  - $\text{Conv2d}(3 \to 64, k=4, s=2, p=1) \to \text{LeakyReLU}(0.2) \to (64 \times 64)$
  - $\text{Conv2d}(64 \to 128, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(128) \to \text{LeakyReLU}(0.2) \to (32 \times 32)$
  - $\text{Conv2d}(128 \to 256, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(256) \to \text{LeakyReLU}(0.2) \to (16 \times 16)$
  - $\text{Conv2d}(256 \to 512, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(512) \to \text{LeakyReLU}(0.2) \to (8 \times 8)$
  - $\text{Conv2d}(512 \to 512, k=4, s=2, p=1, \text{bias}=\text{False}) \to \text{BatchNorm2d}(512) \to \text{LeakyReLU}(0.2) \to (4 \times 4)$
  - $\text{Conv2d}(512 \to 1, k=4, s=1, p=0) \to \text{Sigmoid}() \to 1$
- **Parameter Breakdown:**
  - **Generator Parameters:** $8,055,587$
  - **Discriminator Parameters:** $6,960,961$
  - **Total GAN Parameters:** **$15,016,548$** ($\sim 15.02\text{M}$)

---

## Comprehensive Training & Evaluation Metrics

### Basic GAN Quantitative Results

The Basic Convolutional GAN was trained on **$30,000$ CelebA images** over **$30$ epochs** with a batch size of **$32$**.

#### Summary Evaluation Metrics (`latest_results/gan_evaluation_metrics.csv`)
| Evaluation Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Final Generator Loss** | `4.495222` | Adversarial cross-entropy loss of G attempting to fool D |
| **Minimum Generator Loss** | `3.273938` | Achieved at Epoch 6 during early adversarial balance |
| **Maximum Generator Loss** | `4.533112` | Occurred at Epoch 1 during initial distribution alignment |
| **Final Discriminator Loss** | `0.600166` | Binary cross-entropy loss distinguishing real vs fake |
| **Minimum Discriminator Loss**| `0.600166` | Achieved at Epoch 30 (monotonic convergence of D) |
| **Maximum Discriminator Loss**| `1.000072` | Occurred at Epoch 1 |
| **Real Image Accuracy** | `12.21%` ($0.122070$) | D classification rate on held-out real CelebA images |
| **Fake Image Accuracy** | `99.90%` ($0.999023$) | D classification rate on generated synthetic faces |
| **Overall Discriminator Accuracy**| `56.05%` ($0.560547$)| Combined classification accuracy over 1,000 evaluation images |
| **Generated Pixel Mean** | `-0.136438` | Normalized image tensor mean in range $[-1, 1]$ |
| **Generated Pixel Std** | `0.584781` | Standard deviation showing contrast and color spread |
| **Generated Pixel Range** | `[-1.000000, 0.999996]` | Full utilization of the dynamic output range |

#### Epoch-by-Epoch Loss History (`latest_results/gan_training_history.csv`)
| Epoch | Generator Loss | Discriminator Loss | Epoch | Generator Loss | Discriminator Loss |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.533112 | 1.000072 | **16** | 3.993180 | 0.748097 |
| **2** | 3.541290 | 0.977390 | **17** | 4.029809 | 0.734822 |
| **3** | 3.555026 | 0.958043 | **18** | 3.950045 | 0.742671 |
| **4** | 3.481935 | 0.962400 | **19** | 4.016798 | 0.739464 |
| **5** | 3.328523 | 0.944803 | **20** | 4.031261 | 0.722287 |
| **6** | 3.273938 *(Min)* | 0.939686 | **21** | 4.118321 | 0.710926 |
| **7** | 3.313021 | 0.948323 | **22** | 4.165122 | 0.682024 |
| **8** | 3.414973 | 0.898987 | **23** | 4.192984 | 0.690773 |
| **9** | 3.574835 | 0.879702 | **24** | 4.191601 | 0.657565 |
| **10** | 3.475043 | 0.856819 | **25** | 4.192720 | 0.675889 |
| **11** | 3.598378 | 0.849539 | **26** | 4.339480 | 0.650002 |
| **12** | 3.709502 | 0.832534 | **27** | 4.374483 | 0.654482 |
| **13** | 3.772042 | 0.819086 | **28** | 4.366397 | 0.624995 |
| **14** | 3.818043 | 0.775884 | **29** | 4.444997 | 0.624267 |
| **15** | 3.949660 | 0.767284 | **30** | 4.495222 | 0.600166 *(Min)* |

---

### VAE Quantitative Training Progression

The VAE model was trained using a compound objective consisting of **Reconstruction Loss (MSE)**, **KL Divergence Regularization**, and **Adversarial / Feature Matching assistance** over recorded epochs.

#### Loss Progression Over Recorded Training Epochs
| Epoch | Generator / VAE Loss | Reconstruction Loss (MSE) | KL Loss | Adversarial Loss | Feature Matching Loss | Discriminator Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.9231 | 0.2224 | 8.2977 | 16.5117 | 0.5217 | 0.5131 |
| **2** | 4.4828 | 0.2256 | 10.6285 | 13.4931 | 0.4334 | 0.5258 |
| **3** | 4.2447 | 0.2158 | 10.5480 | 12.4886 | 0.4112 | 0.4951 |
| **4** | 4.0688 | 0.2111 | 10.2509 | 11.6485 | 0.3864 | 0.4656 |
| **5** | 4.0624 | 0.2104 | 9.3502 | 11.6277 | 0.3861 | 0.4552 |
| **6** | 3.9232 | 0.2027 | 9.0026 | 11.5145 | 0.3588 | 0.4452 |
| **7** | 3.8117 | 0.1952 | 8.5454 | 11.2795 | 0.3511 | 0.4272 |
| **8** | 3.7684 | 0.1924 | 9.0033 | 11.4196 | 0.3332 | 0.4319 |
| **9** | 3.7255 | 0.1897 | 8.6359 | 11.2437 | 0.3326 | 0.4212 |
| **10** | **3.6414** | **0.1843** | **8.3643** | **11.3064** | **0.3132** | **0.4241** |

#### Key VAE Quantitative Takeaways:
- **Reconstruction Improvement:** The reconstruction loss decreased from $0.2224$ to $0.1843$, representing an approximate **$17.13\%$ relative improvement** in facial structural reconstruction accuracy.
- **Feature Matching Loss:** Feature-space distance dropped monotonically from $0.5217$ to $0.3132$ (**$39.97\%$ decrease**), demonstrating that decoded faces converged closer to real faces in deep feature representations.
- **KL Regularization:** KL divergence settled near $\sim 8.36$, ensuring that the $64$-D latent space remained compact, continuous, and suitable for smooth random sampling.

---

## Model Comparison Matrix

| Dimension | Variational Autoencoder (VAE) | Basic Convolutional GAN |
| :--- | :--- | :--- |
| **Architectural Family** | Encoder-Decoder with probabilistic bottleneck | Minimax Generator-Discriminator game |
| **Latent Space** | $64$-D continuous Gaussian distribution | $128$-D Standard normal prior $\mathcal{N}(0, I)$ |
| **Native Output Resolution** | $64 \times 64 \times 3$ | $128 \times 128 \times 3$ |
| **Presentation Resolution** | $128 \times 128$ *(via high-quality Bicubic upscaling)* | $128 \times 128$ *(native)* |
| **Total Learnable Parameters** | $2,172,099$ ($1.22\text{M}$ Enc + $0.96\text{M}$ Dec) | $15,016,548$ ($8.06\text{M}$ Gen + $6.96\text{M}$ Disc) |
| **Training Stability** | Monotonically stable; objective converges smoothly | Dynamic adversarial equilibrium; prone to D dominance |
| **Mode Dropping / Collapse** | Resistant; KL regularization forces manifold coverage | Vulnerable; Generator focuses on high-confidence modes |
| **Observed Visual Quality** | Smooth, soft textures, slight blur, high structural symmetry | Crisp edge transitions, higher contrast, localized artifacts |
| **Sampling Mechanism** | $z \sim \mathcal{N}(\mu_{\text{learned}}, \sigma_{\text{learned}}^2) \to \text{Decoder}$ | $z \sim \mathcal{N}(0, I_{128}) \to \text{Generator}$ |

---

## Qualitative Analysis & Observations

### 1. The Blur vs. Sharpness Trade-Off
- **VAE Outputs:** Pixel-space objectives (e.g., MSE) penalize the expected deviation over all modes, driving the decoder toward a visual "average" of possible facial textures. Consequently, VAE-generated faces are clean and anatomically coherent, but softer in high-frequency detail (e.g., individual hair strands and skin pores).
- **Basic GAN Outputs:** The discriminator specifically penalizes blurry or unrealistic transitions, forcing the generator to synthesize sharp edges and high-contrast facial contours. However, without advanced stabilization mechanisms (such as spectral normalization or self-attention), slight structural asymmetries (e.g., ear or boundary distortions) occasionally emerge.

### 2. Critical Interpretation of Discriminator Accuracy
- In `latest_results/gan_evaluation_metrics.csv`, the discriminator achieves **$99.90\%$ fake image accuracy** and **$12.21\%$ real image accuracy** ($56.05\%$ overall).
- **Academic Note:** Discriminator accuracy must **NOT** be interpreted as a direct measure of human perceptual image quality. In standard GAN training, high fake detection accuracy typically signals **discriminator dominance**, where the discriminator outpaces the generator in feature space.

---

## Installation & Execution

### 1. Prerequisites
Ensure you have Python 3.10 or 3.11 installed.

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/Shailesh-S-04/Face-Generation-Using-GAN-VAE.git
cd Face-Generation-Using-GAN-VAE

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Interactive Application
```bash
streamlit run app.py
```
*The application automatically detects whether CUDA hardware acceleration is available (e.g., NVIDIA GeForce RTX 3050 Laptop GPU) and falls back gracefully to CPU inference if unavailable.*

---

## Streamlit Interactive Web App Structure

The upgraded application provides **6 comprehensive presentation tabs**:

1. **Home / Project Overview**: Motivation, CelebA dataset description, KPI metric cards, and live generation previews from both models.
2. **VAE Face Generation**: Interactive face generation ($1$ to $36$ faces), random seed selection, learned latent distribution toggle ($\mu_{\text{learned}}, \sigma_{\text{learned}}$), latent walk interpolation, image reconstruction, and bulk ZIP downloads.
3. **Basic GAN Face Generation**: Native $128 \times 128$ face generation, random seed controls, live Discriminator confidence scoring ($D(G(z))$), and bulk ZIP downloads.
4. **Model Comparison**: Synchronized side-by-side generation test, structural comparison matrix, and qualitative trade-off analysis.
5. **GAN Training & Evaluation**: Dynamic metric cards, interactive 30-epoch loss curves, evaluation grid image viewer, and formal observations.
6. **About / Technical Details**: Detailed layer-by-layer architectural blueprints, dynamic parameter counts, and model file integrity status table.

---

## Authors & Citation
- **Project:** Face Generation Using GAN & VAE
- **Dataset:** CelebA (Large-scale CelebFaces Attributes Dataset)
- **Primary Architectures:** Variational Autoencoder (VAE) & Deep Convolutional Generative Adversarial Network (DCGAN)