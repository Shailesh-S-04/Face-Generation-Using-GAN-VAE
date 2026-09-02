# AI Face Generation using Basic VAE-GAN

This is a Streamlit web application that demonstrates face generation and reconstruction using a trained Basic VAE-GAN model. The model was trained on the CelebA dataset.

## Features
- **Generate Faces**: Sample random latent vectors and generate new synthetic faces.
- **Learned Latent Distribution**: Generate faces using the learned latent distribution (mean and std) from training for better quality.
- **Reconstruct Faces**: Upload a face image to see how the VAE encodes and reconstructs it.
- **Latent Space Interpolation**: Interpolate between two random latent vectors to observe smooth semantic transitions in the generated faces.
- **Download**: Download generated faces individually or as a bulk ZIP file.

## Project Structure
```
project/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── models/                    # Trained model weights (Required)
    ├── encoder_final.pth
    ├── decoder_final.pth
    ├── discriminator_final.pth
    └── latent_distribution.pth
```

## Model Architecture
The architecture is a basic VAE-GAN:
- **VAE (Variational Autoencoder)**: The encoder processes 64x64 RGB images into a 64-dimensional latent space (mean and log variance). The decoder reconstructs faces from these latent vectors.
- **GAN (Generative Adversarial Network)**: The discriminator distinguishes real faces from generated ones, helping the generator (encoder+decoder) produce more realistic outputs.

## Setup and Installation
1. Ensure you have Python 3.10 or 3.11 installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the four trained `.pth` model files are placed inside the `models/` directory.

## Running the Application
To start the Streamlit application, run the following command in your terminal:
```bash
streamlit run app.py
```
This will open the application in your default web browser.

## Device Compatibility
The application automatically detects whether an NVIDIA GPU (CUDA) is available. If a GPU is not found, it will seamlessly fall back to CPU inference.