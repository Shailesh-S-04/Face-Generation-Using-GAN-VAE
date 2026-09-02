import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import zipfile
import os
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Basic VAE-GAN Face Gen", layout="wide")

IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Model Architectures (Exactly as specified)
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        flat_dim = 256 * (64 // 16) ** 2

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return (
            self.fc_mu(h),
            self.fc_logvar(h)
        )

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.init_size = 64 // 16
        self.fc = nn.Linear(latent_dim, 256 * self.init_size ** 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 256, self.init_size, self.init_size)
        return self.decoder(h)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        flat_dim = 256 * (64 // 16) ** 2
        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        out = self.classifier(feat.flatten(1))
        if return_features:
            return out, feat
        return out


# -----------------------------------------------------------------------------
# Loading Functions
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Models...")
def load_models():
    models_dir = "models"
    encoder_path = os.path.join(models_dir, "encoder_final.pth")
    decoder_path = os.path.join(models_dir, "decoder_final.pth")
    latent_path = os.path.join(models_dir, "latent_distribution.pth")

    missing = []
    for path in [encoder_path, decoder_path, latent_path]:
        if not os.path.exists(path):
            missing.append(path)
            
    if missing:
        return None, None, None, f"Model file(s) not found: {', '.join(missing)}. Please place the trained .pth files inside the models/ folder."

    try:
        encoder = Encoder(LATENT_DIM).to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        encoder.eval()

        decoder = Decoder(LATENT_DIM).to(device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        decoder.eval()

        # Load latent distribution cautiously
        latent_dist_raw = torch.load(latent_path, map_location=device, weights_only=True)
        latent_dist = None
        if isinstance(latent_dist_raw, dict) and 'mean' in latent_dist_raw and 'std' in latent_dist_raw:
            latent_dist = {
                'mean': latent_dist_raw['mean'].to(device),
                'std': latent_dist_raw['std'].to(device)
            }
        
        return encoder, decoder, latent_dist, None
    except Exception as e:
        return None, None, None, f"Architecture and checkpoint are incompatible or another loading error occurred: {str(e)}"

encoder, decoder, latent_dist, load_error = load_models()

# -----------------------------------------------------------------------------
# Preprocessing & Helpers
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def tensor_to_pil(tensor):
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Generation Controls")
num_images = st.sidebar.slider("Number of faces", min_value=1, max_value=32, value=8)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
use_learned_latent = st.sidebar.checkbox("Use learned latent distribution", value=True)

if st.sidebar.button("Generate Faces"):
    st.session_state["generate_clicked"] = True

st.sidebar.divider()
st.sidebar.subheader("Model Status")
if load_error:
    st.sidebar.error(load_error)
else:
    st.sidebar.success("Encoder: Loaded ✓")
    st.sidebar.success("Decoder: Loaded ✓")
    if latent_dist:
        st.sidebar.success("Latent Distribution: Loaded ✓")
    else:
        st.sidebar.warning("Latent Distribution: Not loaded")
    st.sidebar.info("Discriminator: Not loaded — Training only")
st.sidebar.info(f"Device: {device.type.upper()}")

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("AI Face Generation using Basic VAE-GAN")
st.markdown("Generate synthetic human faces using a Variational Autoencoder and Generative Adversarial Network.")

st.info("**Model Type:** Basic VAE-GAN | **Dataset:** CelebA | **Image Resolution:** 64 × 64 | **Latent Dimension:** 64 | **Framework:** PyTorch")

if load_error:
    st.error(load_error)
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Generate Faces", "Reconstruct Face", "Latent Space", "Training Results", "About Model"])

with tab1:
    st.header("Generate New Faces")
    st.write("Generate synthetic faces from the learned latent space.")
    
    if st.button("Generate Again"):
        st.session_state["generate_clicked"] = True
        
    if st.session_state.get("generate_clicked", False):
        torch.manual_seed(random_seed)
        
        with torch.no_grad():
            if use_learned_latent and latent_dist:
                mean = latent_dist['mean']
                std = latent_dist['std']
                z = mean + std * torch.randn(num_images, LATENT_DIM, device=device)
            else:
                z = torch.randn(num_images, LATENT_DIM, device=device)
            
            generated = decoder(z)
            
        cols_per_row = 8 if num_images > 8 else 4
        
        images_to_download = []
        
        # Display in grid
        cols = st.columns(cols_per_row)
        for i in range(num_images):
            img = tensor_to_pil(generated[i])
            images_to_download.append((f"generated_face_{i+1}.png", img))
            with cols[i % cols_per_row]:
                st.image(img, caption=f"Generated Face {i+1}", use_container_width=True)
                
        st.divider()
        st.subheader("Download Generated Faces")
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_name, img in images_to_download:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                zip_file.writestr(file_name, img_byte_arr.getvalue())
        
        st.download_button(
            label="Download All as ZIP",
            data=zip_buffer.getvalue(),
            file_name="generated_faces.zip",
            mime="application/zip"
        )

with tab2:
    st.header("VAE Face Reconstruction")
    st.write("Upload a face image to see how the VAE encodes and reconstructs it.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, logvar = encoder(input_tensor)
            # Reparameterization trick for deterministic reconstruction: we can just use mu
            z = mu
            reconstructed_tensor = decoder(z)[0]
            
        reconstructed_img = tensor_to_pil(reconstructed_tensor)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(reconstructed_img, caption="Reconstructed Image", use_container_width=True)

with tab3:
    st.header("Latent Space Exploration")
    st.write("Interpolate between two random latent vectors to observe smooth transitions.")
    
    interp_steps = st.slider("Interpolation Steps", min_value=3, max_value=16, value=8)
    
    if st.button("Generate Interpolation"):
        torch.manual_seed(random_seed + 1) # Ensure it's different but reproducible
        with torch.no_grad():
            if use_learned_latent and latent_dist:
                mean = latent_dist['mean']
                std = latent_dist['std']
                z1 = mean + std * torch.randn(1, LATENT_DIM, device=device)
                z2 = mean + std * torch.randn(1, LATENT_DIM, device=device)
            else:
                z1 = torch.randn(1, LATENT_DIM, device=device)
                z2 = torch.randn(1, LATENT_DIM, device=device)
            
            alphas = torch.linspace(0, 1, steps=interp_steps).to(device)
            interpolated_z = []
            for alpha in alphas:
                interpolated_z.append((1 - alpha) * z1 + alpha * z2)
            
            interpolated_z = torch.cat(interpolated_z, dim=0)
            generated_interp = decoder(interpolated_z)
            
        st.write("### VAE Latent Space Interpolation")
        cols = st.columns(interp_steps)
        for i in range(interp_steps):
            img = tensor_to_pil(generated_interp[i])
            with cols[i]:
                if i == 0:
                    caption = "Face A"
                elif i == interp_steps - 1:
                    caption = "Face B"
                else:
                    caption = f"Step {i}"
                st.image(img, caption=caption, use_container_width=True)

with tab4:
    st.header("Training Results")
    st.write("Visualizing the recorded loss metrics from the Kaggle VAE-GAN training over 60 epochs.")
    
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gen_loss = [4.9231, 4.4828, 4.2447, 4.0688, 4.0624, 3.9232, 3.8117, 3.7684, 3.7255, 3.6414]
    disc_loss = [0.5131, 0.5258, 0.4951, 0.4656, 0.4552, 0.4452, 0.4272, 0.4319, 0.4212, 0.4241]
    recon_loss = [0.2224, 0.2256, 0.2158, 0.2111, 0.2104, 0.2027, 0.1952, 0.1924, 0.1897, 0.1843]
    kl_loss = [8.2977, 10.6285, 10.5480, 10.2509, 9.3502, 9.0026, 8.5454, 9.0033, 8.6359, 8.3643]
    adv_loss = [16.5117, 13.4931, 12.4886, 11.6485, 11.6277, 11.5145, 11.2795, 11.4196, 11.2437, 11.3064]
    feat_loss = [0.5217, 0.4334, 0.4112, 0.3864, 0.3861, 0.3588, 0.3511, 0.3332, 0.3326, 0.3132]

    st.subheader("Training Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Epochs", 10)
        recon_improvement = ((recon_loss[0] - recon_loss[-1]) / recon_loss[0]) * 100
        st.metric("Reconstruction Improvement", f"approx. {recon_improvement:.1f}%")
    with col2:
        st.metric("Initial Gen/VAE Loss", f"{gen_loss[0]:.4f}")
        st.metric("Final Gen/VAE Loss", f"{gen_loss[-1]:.4f}")
    with col3:
        st.metric("Initial Recon Loss", f"{recon_loss[0]:.4f}")
        st.metric("Final Recon Loss", f"{recon_loss[-1]:.4f}")
    with col4:
        st.metric("Initial Disc Loss", f"{disc_loss[0]:.4f}")
        st.metric("Final Disc Loss", f"{disc_loss[-1]:.4f}")
        
    st.divider()
    
    st.subheader("A. VAE-GAN Training Loss")
    df_train = pd.DataFrame({"Epoch": epochs, "Generator / VAE Loss": gen_loss, "Discriminator Loss": disc_loss}).set_index("Epoch")
    st.line_chart(df_train)
    
    st.subheader("B. Reconstruction Loss During VAE-GAN Training")
    df_recon = pd.DataFrame({"Epoch": epochs, "Reconstruction Loss": recon_loss}).set_index("Epoch")
    st.line_chart(df_recon)
    st.caption("Lower reconstruction loss indicates that the decoder is becoming better at reproducing the structure of the input face images.")
    
    st.subheader("C. VAE-GAN Component Losses")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Chart 3A: KL Loss**")
        df_kl = pd.DataFrame({"Epoch": epochs, "KL Loss": kl_loss}).set_index("Epoch")
        st.line_chart(df_kl)
    with col_b:
        st.markdown("**Chart 3B: Adversarial Loss**")
        df_adv = pd.DataFrame({"Epoch": epochs, "Adversarial Loss": adv_loss}).set_index("Epoch")
        st.line_chart(df_adv)
    with col_c:
        st.markdown("**Chart 3C: Feature Matching Loss**")
        df_feat = pd.DataFrame({"Epoch": epochs, "Feature Matching Loss": feat_loss}).set_index("Epoch")
        st.line_chart(df_feat)
        
    st.divider()
    
    st.subheader("What do these graphs indicate?")
    st.markdown("""
    1. **Generator/VAE Loss**: "The decreasing generator/VAE loss indicates that the encoder-decoder and adversarial objectives are improving during training."
    2. **Discriminator Loss**: "The discriminator loss remains relatively low and stable, indicating that the discriminator is successfully distinguishing real and generated images during training."
    3. **Reconstruction Loss**: "The reconstruction loss decreases from 0.2224 to 0.1843, showing that the decoder learns to reproduce facial structure more accurately."
    4. **Feature Matching Loss**: "The decreasing feature matching loss indicates that generated/reconstructed images are becoming closer to real images in the discriminator's learned feature space."
    5. **KL Loss**: "KL loss represents the regularization of the learned latent distribution so that the latent space remains structured and useful for generation."
    6. **Adversarial Loss**: "The adversarial loss represents the generator's effort to produce images that the discriminator considers realistic."
    """)
    
    st.divider()
    
    st.subheader("Model Limitations")
    st.markdown("The current model is a basic VAE-GAN trained on 64×64 face images. Because of the limited resolution, training duration, and relatively simple architecture, some generated faces contain blur, artifacts, or distorted facial features. The model successfully demonstrates the complete VAE-GAN generation pipeline and learned latent representation, but it is not intended to achieve photorealistic face synthesis.")

with tab5:
    with st.expander("About the Model", expanded=True):
        st.markdown("""
        ### Architecture Overview
        This project uses a basic VAE-GAN architecture.

        **VAE (Variational Autoencoder):**
        - The encoder maps an input face into a probabilistic latent representation described by mean and variance.
        - The decoder reconstructs or generates faces from latent vectors.

        **GAN (Generative Adversarial Network):**
        - The discriminator learns to distinguish real CelebA faces from generated/reconstructed faces.
        - The generator side (encoder + decoder) learns to produce increasingly realistic faces.
        
        *Note: The Discriminator was used during training to improve generation quality, but it is not required during inference. Therefore, it is intentionally bypassed in this application.*
        
        ---
        
        ### Face Generation Workflow
        ```text
        Random Latent Vector
                ↓
           VAE Decoder
                ↓
         Generated Face
        ```
        
        ### Face Reconstruction Workflow
        ```text
        Input Face
            ↓
         Encoder
            ↓
        Latent Space
            ↓
         Decoder
            ↓
        Reconstructed Face
        ```
        
        ### Training Diagram (Reference)
        ```text
        Real Face
           ↓
        Encoder → Latent Space → Decoder
                                 ↓
                          Generated Face
                                 ↓
                          Discriminator
                                 ↑
                            Real Faces
        ```
        """)
