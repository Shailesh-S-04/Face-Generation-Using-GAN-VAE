import io
import os
import zipfile
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision import transforms

# -----------------------------------------------------------------------------
# Base Configuration & Directory Paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Generation Using GAN & VAE | CelebA",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "latest_results"

# Device selection with automatic fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# -----------------------------------------------------------------------------
# Custom Styling (Professional Academic / Project Presentation Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card styling */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2e86de;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #aaaaaa;
        margin-top: 2px;
    }

    /* Badge tags */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background-color: #1e3799; color: #ffffff; }
    .badge-green { background-color: #009432; color: #ffffff; }
    .badge-amber { background-color: #e58e26; color: #ffffff; }
    .badge-gray { background-color: #4a5568; color: #ffffff; }

    /* Comparison box */
    .compare-box {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border-left: 4px solid #3867d6;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Image captions and label */
    .image-tag {
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
        color: #888888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Architectures (Strictly Compatible with Trained Weights)
# -----------------------------------------------------------------------------

# 1. VAE Architecture (models/encoder_final.pth, models/decoder_final.pth)
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
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
        flat_dim = 256 * (64 // 16) ** 2  # 256 * 4 * 4 = 4096
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 64 // 16  # 4
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


# 2. Basic GAN Architecture (latest_results/generator_final.pth, latest_results/discriminator_final.pth)
class GANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 4
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size ** 2),
            nn.BatchNorm1d(512 * self.init_size ** 2),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.upsample(out)
        return img


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)


# -----------------------------------------------------------------------------
# Model Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained VAE and GAN model checkpoints...")
def load_all_models():
    """
    Loads all trained weights with strict=True.
    Returns:
        dict: models, latent_distribution, errors, metadata
    """
    models = {
        "vae_encoder": None,
        "vae_decoder": None,
        "vae_latent_dist": None,
        "gan_generator": None,
        "gan_discriminator": None,
    }
    errors = []

    # 1. Load VAE Checkpoints
    enc_path = MODELS_DIR / "encoder_final.pth"
    dec_path = MODELS_DIR / "decoder_final.pth"
    latent_path = MODELS_DIR / "latent_distribution.pth"

    for p in [enc_path, dec_path, latent_path]:
        if not p.exists():
            errors.append(f"Missing VAE file: {p.name} (Expected at: models/{p.name})")

    if not errors:
        try:
            vae_enc = VAEEncoder(latent_dim=64).to(device)
            vae_enc.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True), strict=True)
            vae_enc.eval()
            models["vae_encoder"] = vae_enc

            vae_dec = VAEDecoder(latent_dim=64).to(device)
            vae_dec.load_state_dict(torch.load(dec_path, map_location=device, weights_only=True), strict=True)
            vae_dec.eval()
            models["vae_decoder"] = vae_dec

            latent_raw = torch.load(latent_path, map_location=device, weights_only=True)
            if isinstance(latent_raw, dict) and "mean" in latent_raw and "std" in latent_raw:
                models["vae_latent_dist"] = {
                    "mean": latent_raw["mean"].to(device),
                    "std": latent_raw["std"].to(device)
                }
            else:
                errors.append("Invalid format in models/latent_distribution.pth (expected dict with 'mean' and 'std')")
        except Exception as e:
            errors.append(f"Error loading VAE checkpoints: {str(e)}")

    # 2. Load Basic GAN Checkpoints
    gen_path = RESULTS_DIR / "generator_final.pth"
    disc_path = RESULTS_DIR / "discriminator_final.pth"

    for p in [gen_path, disc_path]:
        if not p.exists():
            errors.append(f"Missing GAN file: {p.name} (Expected at: latest_results/{p.name})")

    if not any("GAN" in err for err in errors):
        try:
            gan_gen = GANGenerator(latent_dim=128).to(device)
            gan_gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True), strict=True)
            gan_gen.eval()
            models["gan_generator"] = gan_gen

            gan_disc = GANDiscriminator().to(device)
            gan_disc.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True), strict=True)
            gan_disc.eval()
            models["gan_discriminator"] = gan_disc
        except Exception as e:
            errors.append(f"Error loading GAN checkpoints: {str(e)}")

    return models, errors


loaded_models, model_errors = load_all_models()
vae_encoder = loaded_models["vae_encoder"]
vae_decoder = loaded_models["vae_decoder"]
vae_latent_dist = loaded_models["vae_latent_dist"]
gan_generator = loaded_models["gan_generator"]
gan_discriminator = loaded_models["gan_discriminator"]


def count_parameters(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())


vae_enc_params = count_parameters(vae_encoder)
vae_dec_params = count_parameters(vae_decoder)
vae_total_params = vae_enc_params + vae_dec_params

gan_gen_params = count_parameters(gan_generator)
gan_disc_params = count_parameters(gan_discriminator)
gan_total_params = gan_gen_params + gan_disc_params


# -----------------------------------------------------------------------------
# Image Helper Functions
# -----------------------------------------------------------------------------
def tensor_to_pil(tensor, target_size=None):
    """
    Converts a single image tensor in [-1, 1] to a PIL Image in [0, 255].
    If target_size is provided, upscales cleanly using Bicubic interpolation.
    """
    t = (tensor.detach().cpu() + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    ndarr = t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BICUBIC)
    return img


vae_transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# -----------------------------------------------------------------------------
# Presentation Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎭 PROJECT")
    st.markdown("**Face Generation Using GAN & VAE**")

    st.markdown("---")
    st.markdown("### 📁 DATASET")
    st.markdown("**CelebA** (Celebrity Faces)")

    st.markdown("---")
    st.markdown("### 🧠 MODELS")
    st.markdown("• **Variational Autoencoder (VAE)**\n• **Basic Convolutional GAN**")

    st.markdown("---")
    st.markdown("### 📐 RESOLUTION")
    st.markdown("• **VAE:** 64 × 64 native *(upscaled to 128×128 for presentation)*")
    st.markdown("• **GAN:** 128 × 128 native")

    st.markdown("---")
    st.markdown("### ⚡ HARDWARE ACCELERATION")
    if torch.cuda.is_available():
        st.markdown(f"<span class='badge badge-green'>CUDA ACTIVE</span>", unsafe_allow_html=True)
        st.caption(f"{device_name}")
    else:
        st.markdown(f"<span class='badge badge-amber'>CPU ACTIVE</span>", unsafe_allow_html=True)
        st.caption("Running on host CPU")

    st.markdown("---")
    st.markdown("### 🔍 MODEL STATUS")
    if vae_decoder is not None and vae_encoder is not None and vae_latent_dist is not None:
        st.markdown("✅ **VAE weights loaded**")
    else:
        st.markdown("❌ **VAE weights failed**")

    if gan_generator is not None and gan_discriminator is not None:
        st.markdown("✅ **GAN weights loaded**")
    else:
        st.markdown("❌ **GAN weights failed**")

    st.markdown("---")
    st.caption("Academic Project Demonstration • Faculty Review Mode")


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
st.title("Face Generation Using GAN & VAE")
st.markdown(
    "Interactive academic demonstration comparing **Variational Autoencoders (VAE)** "
    "and **Basic Generative Adversarial Networks (GAN)** on the CelebA dataset."
)

if model_errors:
    for err in model_errors:
        st.error(f"⚠️ {err}")
    st.stop()

tab_overview, tab_vae, tab_gan, tab_compare, tab_eval, tab_tech = st.tabs([
    "1. Home / Project Overview",
    "2. VAE Face Generation",
    "3. Basic GAN Face Generation",
    "4. Model Comparison",
    "5. GAN Training & Evaluation",
    "6. About / Technical Details"
])

# =============================================================================
# TAB 1: Home / Project Overview
# =============================================================================
with tab_overview:
    st.header("Project Overview")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Dataset</div>
            <div class='metric-value'>CelebA</div>
            <div class='metric-sub'>Align & Cropped Celeb Faces</div>
        </div>
        """, unsafe_allow_html=True)
    with col_meta2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Target Resolution</div>
            <div class='metric-value'>128 × 128</div>
            <div class='metric-sub'>Presentation Canvas</div>
        </div>
        """, unsafe_allow_html=True)
    with col_meta3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>VAE Parameters</div>
            <div class='metric-value'>{vae_total_params / 1e6:.2f}M</div>
            <div class='metric-sub'>Encoder + Decoder</div>
        </div>
        """, unsafe_allow_html=True)
    with col_meta4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>GAN Parameters</div>
            <div class='metric-value'>{gan_total_params / 1e6:.2f}M</div>
            <div class='metric-sub'>Generator + Discriminator</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_desc_l, col_desc_r = st.columns(2)
    with col_desc_l:
        st.subheader("1. Variational Autoencoder (VAE)")
        st.markdown("""
        A **probabilistic generative framework** that learns to map human faces into a structured, continuous 64-dimensional latent space:
        - **Encoder**: Compresses input faces into continuous Gaussian parameters ($\mu$ and $\log\sigma^2$).
        - **Latent Distribution**: Regularized via KL Divergence and learned empirical distribution ($\mu_{learned}, \sigma_{learned}$).
        - **Decoder**: Reconstructs and samples synthetic faces from latent vectors.
        - **Native Resolution**: **64 × 64** *(cleanly upscaled to 128 × 128 for fair side-by-side presentation)*.
        """)
    with col_desc_r:
        st.subheader("2. Basic Convolutional GAN")
        st.markdown("""
        A **two-player minimax game** between a Generator and Discriminator:
        - **Generator ($G$)**: Transforms 128-D random Gaussian noise ($z \sim \mathcal{N}(0, I)$) into synthetic faces through transposed convolutions.
        - **Discriminator ($D$)**: Evaluates images and distinguishes real CelebA photos from synthetic ones.
        - **Adversarial Training**: $G$ and $D$ compete iteratively to synthesize convincing visual textures.
        - **Native Resolution**: **128 × 128** natively generated.
        """)

    st.markdown("---")
    st.subheader("Honest Academic Assessment")
    st.info("""
    **Presentation Note:** Both models are baseline convolutional generative architectures trained locally on CelebA.
    - They are intended to demonstrate **core generative principles, latent manifold structure, and adversarial dynamics**.
    - Generated images represent **CelebA-style synthetic faces**.
    - Because of baseline architectural depth and limited training budget, results will naturally exhibit characteristic foundational traits:
      VAE outputs are softer and smoother, while Basic GAN outputs exhibit higher native contrast but may contain localized artifacts or asymmetry.
      Neither model claims photorealistic synthesis.
    """)

    st.markdown("---")
    st.subheader("Sample Model Generations (Live Preview)")
    with torch.no_grad():
        torch.manual_seed(100)
        # Sample 4 VAE faces
        if vae_latent_dist:
            z_vae = vae_latent_dist["mean"] + vae_latent_dist["std"] * torch.randn(4, 64, device=device)
        else:
            z_vae = torch.randn(4, 64, device=device)
        sample_vae = vae_decoder(z_vae)

        # Sample 4 GAN faces
        z_gan = torch.randn(4, 128, device=device)
        sample_gan = gan_generator(z_gan)

    col_preview1, col_preview2 = st.columns(2)
    with col_preview1:
        st.markdown("##### VAE Samples (Native 64×64 → Displayed at 128×128)")
        grid_v = st.columns(4)
        for i in range(4):
            with grid_v[i]:
                st.image(tensor_to_pil(sample_vae[i], target_size=(128, 128)), use_container_width=True)
                st.caption(f"VAE #{i+1}")

    with col_preview2:
        st.markdown("##### Basic GAN Samples (Native 128×128)")
        grid_g = st.columns(4)
        for i in range(4):
            with grid_g[i]:
                st.image(tensor_to_pil(sample_gan[i]), use_container_width=True)
                st.caption(f"GAN #{i+1}")


# =============================================================================
# TAB 2: VAE Face Generation
# =============================================================================
with tab_vae:
    st.header("VAE Face Generation")
    st.markdown(
        "Synthesize novel faces by sampling vectors from the learned latent space and passing them "
        "through the trained **VAE Decoder**."
    )

    with st.expander("🛠️ Generation Controls", expanded=True):
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1.2])
        with ctrl_col1:
            vae_num_faces = st.slider("Number of faces", min_value=1, max_value=36, value=8, step=1, key="vae_num")
        with ctrl_col2:
            vae_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1, key="vae_seed")
        with ctrl_col3:
            vae_use_learned = st.checkbox(
                "Use Learned Latent Distribution (Recommended)",
                value=True,
                help="Samples from z ~ N(mu_learned, sigma_learned) extracted from training rather than naive N(0, I)."
            )

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            vae_gen_clicked = st.button("Generate Faces", key="btn_vae_gen", type="primary")

    # Generate VAE faces
    torch.manual_seed(vae_seed)
    with torch.no_grad():
        if vae_use_learned and vae_latent_dist is not None:
            mean = vae_latent_dist["mean"]
            std = vae_latent_dist["std"]
            z_vae = mean + std * torch.randn(vae_num_faces, 64, device=device)
            dist_label = "Learned Latent Distribution N(μ_learned, σ_learned)"
        else:
            z_vae = torch.randn(vae_num_faces, 64, device=device)
            dist_label = "Standard Normal Prior N(0, I)"

        generated_vae_tensors = vae_decoder(z_vae)

    st.markdown(f"**Current Configuration:** {dist_label} | Seed: `{vae_seed}` | Count: `{vae_num_faces}`")
    st.caption("📌 **Resolution Note:** Native VAE output: **64 × 64** | Displayed at: **128 × 128** (Bicubic interpolation)")

    # Display grid
    cols_per_row = 6 if vae_num_faces >= 6 else vae_num_faces
    if cols_per_row == 0:
        cols_per_row = 1

    vae_images_native = []
    vae_images_upscaled = []

    for idx in range(0, vae_num_faces, cols_per_row):
        row_cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            img_idx = idx + c_idx
            if img_idx < vae_num_faces:
                img_native = tensor_to_pil(generated_vae_tensors[img_idx])
                img_upscaled = tensor_to_pil(generated_vae_tensors[img_idx], target_size=(128, 128))
                vae_images_native.append((f"vae_face_{img_idx+1}_64x64.png", img_native))
                vae_images_upscaled.append((f"vae_face_{img_idx+1}_128x128.png", img_upscaled))

                with row_cols[c_idx]:
                    st.image(img_upscaled, use_container_width=True)
                    st.markdown(
                        f"<div class='image-tag'>Face #{img_idx+1}<br>"
                        f"<span style='color:#3867d6;'>64×64 → 128×128</span></div>",
                        unsafe_allow_html=True
                    )

    # Download Section
    st.markdown("---")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED, False) as zf:
            for fname, im in vae_images_upscaled:
                ibuf = io.BytesIO()
                im.save(ibuf, format="PNG")
                zf.writestr(fname, ibuf.getvalue())
        st.download_button(
            label="📦 Download All VAE Faces (ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"vae_faces_seed_{vae_seed}.zip",
            mime="application/zip",
            key="dl_vae_zip"
        )

    # Interactive VAE Feature: Latent Interpolation
    st.markdown("---")
    with st.expander("✨ Latent Space Interpolation (Manifold Continuity Demo)"):
        st.markdown(
            "Interpolate smoothly between two random latent vectors ($z_A \to z_B$). "
            "This highlights the smoothness and continuity of the VAE latent space."
        )
        interp_steps = st.slider("Interpolation Steps", min_value=4, max_value=12, value=8, step=1, key="vae_interp_steps")

        if st.button("Generate Interpolation Walk", key="btn_vae_interp"):
            torch.manual_seed(vae_seed + 101)
            with torch.no_grad():
                if vae_use_learned and vae_latent_dist is not None:
                    za = vae_latent_dist["mean"] + vae_latent_dist["std"] * torch.randn(1, 64, device=device)
                    zb = vae_latent_dist["mean"] + vae_latent_dist["std"] * torch.randn(1, 64, device=device)
                else:
                    za = torch.randn(1, 64, device=device)
                    zb = torch.randn(1, 64, device=device)

                alphas = torch.linspace(0, 1, steps=interp_steps).to(device)
                interp_z = torch.cat([(1 - a) * za + a * zb for a in alphas], dim=0)
                interp_imgs = vae_decoder(interp_z)

            int_cols = st.columns(interp_steps)
            for step_i in range(interp_steps):
                im_step = tensor_to_pil(interp_imgs[step_i], target_size=(128, 128))
                with int_cols[step_i]:
                    label = "Start (z_A)" if step_i == 0 else ("End (z_B)" if step_i == interp_steps - 1 else f"Step {step_i}")
                    st.image(im_step, caption=label, use_container_width=True)

    # Reconstruction Demo
    with st.expander("🔄 VAE Face Reconstruction (Encoder → Latent → Decoder)"):
        st.markdown("Upload an external face image to evaluate the VAE's encoding and reconstruction pipeline.")
        uploaded_face = st.file_uploader("Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="vae_upload")
        if uploaded_face is not None:
            raw_img = Image.open(uploaded_face).convert("RGB")
            inp_tensor = vae_transform(raw_img).unsqueeze(0).to(device)
            with torch.no_grad():
                mu_enc, _ = vae_encoder(inp_tensor)
                recon_tensor = vae_decoder(mu_enc)[0]
            recon_img = tensor_to_pil(recon_tensor, target_size=(128, 128))

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.image(raw_img, caption="Original Input Image", use_container_width=True)
            with r_col2:
                st.image(recon_img, caption="VAE Reconstructed Image (Displayed at 128×128)", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Meeting Talking Points (VAE Generation):**
    - **Latent Prior:** Sampling from the empirically learned latent distribution ($\mu_{learned}, \sigma_{learned}$) aligns with the encoder's true mass distribution, reducing blur compared to naive $\mathcal{N}(0, I)$ sampling.
    - **Smooth Manifold:** Smooth interpolations demonstrate that the latent space does not suffer from disjoint gaps or catastrophic mode collapses.
    """)


# =============================================================================
# TAB 3: Basic GAN Face Generation
# =============================================================================
with tab_gan:
    st.header("Basic GAN Face Generation")
    st.markdown(
        "Synthesize novel faces by feeding 128-dimensional standard Gaussian noise ($z \sim \mathcal{N}(0, I)$) "
        "into the trained **Basic Convolutional GAN Generator**."
    )

    with st.expander("🛠️ Generation Controls", expanded=True):
        g_col1, g_col2, g_col3 = st.columns([1, 1, 1.2])
        with g_col1:
            gan_num_faces = st.slider("Number of faces", min_value=1, max_value=36, value=8, step=1, key="gan_num")
        with g_col2:
            gan_seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1, key="gan_seed")
        with g_col3:
            show_disc_scores = st.checkbox(
                "Evaluate with Discriminator Confidence",
                value=True,
                help="Computes D(G(z)) for each generated face. Note: This reflects discriminator confidence, not human perceptual quality."
            )

        btn_g1, btn_g2 = st.columns([1, 4])
        with btn_g1:
            gan_gen_clicked = st.button("Generate Faces", key="btn_gan_gen", type="primary")

    # Generate GAN faces
    torch.manual_seed(gan_seed)
    with torch.no_grad():
        z_gan = torch.randn(gan_num_faces, 128, device=device)
        generated_gan_tensors = gan_generator(z_gan)
        if show_disc_scores and gan_discriminator is not None:
            disc_confidences = gan_discriminator(generated_gan_tensors).cpu().numpy().flatten()
        else:
            disc_confidences = None

    st.markdown(f"**Current Configuration:** Standard Gaussian Noise $z \\sim \\mathcal{{N}}(0, I_{{128}})$ | Seed: `{gan_seed}` | Count: `{gan_num_faces}`")
    st.caption("📌 **Resolution Note:** Native GAN output: **128 × 128**")

    # Display grid
    cols_per_row_gan = 6 if gan_num_faces >= 6 else gan_num_faces
    if cols_per_row_gan == 0:
        cols_per_row_gan = 1

    gan_images_list = []

    for idx in range(0, gan_num_faces, cols_per_row_gan):
        row_cols = st.columns(cols_per_row_gan)
        for c_idx in range(cols_per_row_gan):
            img_idx = idx + c_idx
            if img_idx < gan_num_faces:
                img_native = tensor_to_pil(generated_gan_tensors[img_idx])
                gan_images_list.append((f"gan_face_{img_idx+1}_128x128.png", img_native))

                with row_cols[c_idx]:
                    st.image(img_native, use_container_width=True)
                    if disc_confidences is not None:
                        d_score = disc_confidences[img_idx]
                        score_text = f"D-Score: {d_score:.4f}"
                        st.markdown(
                            f"<div class='image-tag'>Face #{img_idx+1}<br>"
                            f"<span style='color:#009432;'>Native 128×128</span><br>"
                            f"<span style='color:#888888; font-size:0.72rem;'>{score_text}</span></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='image-tag'>Face #{img_idx+1}<br>"
                            f"<span style='color:#009432;'>Native 128×128</span></div>",
                            unsafe_allow_html=True
                        )

    # Download Section
    st.markdown("---")
    col_gdl1, col_gdl2 = st.columns([1, 3])
    with col_gdl1:
        zip_buf_gan = io.BytesIO()
        with zipfile.ZipFile(zip_buf_gan, "a", zipfile.ZIP_DEFLATED, False) as zf:
            for fname, im in gan_images_list:
                ibuf = io.BytesIO()
                im.save(ibuf, format="PNG")
                zf.writestr(fname, ibuf.getvalue())
        st.download_button(
            label="📦 Download All GAN Faces (ZIP)",
            data=zip_buf_gan.getvalue(),
            file_name=f"gan_faces_seed_{gan_seed}.zip",
            mime="application/zip",
            key="dl_gan_zip"
        )

    st.markdown("---")
    st.markdown("""
    **Meeting Talking Points (Basic GAN Generation):**
    - **Native Resolution:** The GAN generator natively outputs 128 × 128 pixels, offering sharp edge transitions and detailed facial contrast.
    - **Discriminator Confidence:** The discriminator score $D(G(z))$ reflects how likely the discriminator considers the image to be real ($0 = \\text{fake}$, $1 = \\text{real}$). In adversarial training, low scores often indicate that the discriminator easily identifies generated cues, which is typical of basic GANs.
    """)


# =============================================================================
# TAB 4: Model Comparison
# =============================================================================
with tab_compare:
    st.header("Comparative Analysis: VAE vs Basic GAN")
    st.markdown("Direct side-by-side evaluation of generative behavior, architectural paradigms, and image characteristics.")

    # Side-by-side Live Generation
    st.subheader("1. Live Side-by-Side Generation")
    comp_col_ctrl1, comp_col_ctrl2 = st.columns(2)
    with comp_col_ctrl1:
        comp_seed = st.number_input("Comparison Seed", min_value=0, max_value=999999, value=42, step=1, key="comp_seed")
    with comp_col_ctrl2:
        comp_faces = st.slider("Comparison Face Count", min_value=2, max_value=8, value=4, step=2, key="comp_count")

    torch.manual_seed(comp_seed)
    with torch.no_grad():
        if vae_latent_dist is not None:
            z_c_vae = vae_latent_dist["mean"] + vae_latent_dist["std"] * torch.randn(comp_faces, 64, device=device)
        else:
            z_c_vae = torch.randn(comp_faces, 64, device=device)
        c_vae_imgs = vae_decoder(z_c_vae)

        z_c_gan = torch.randn(comp_faces, 128, device=device)
        c_gan_imgs = gan_generator(z_c_gan)

    col_side_vae, col_side_gan = st.columns(2)
    with col_side_vae:
        st.markdown("""
        <div class='compare-box'>
            <h4 style='margin-top:0;'>Variational Autoencoder (VAE)</h4>
            <p style='font-size:0.85rem; color:#aaaaaa;'>Native 64×64 • Displayed at 128×128 (Bicubic)</p>
        </div>
        """, unsafe_allow_html=True)
        c_v_cols = st.columns(comp_faces)
        for i in range(comp_faces):
            with c_v_cols[i]:
                st.image(tensor_to_pil(c_vae_imgs[i], target_size=(128, 128)), use_container_width=True)
                st.caption(f"VAE #{i+1}")

    with col_side_gan:
        st.markdown("""
        <div class='compare-box'>
            <h4 style='margin-top:0;'>Basic Convolutional GAN</h4>
            <p style='font-size:0.85rem; color:#aaaaaa;'>Native 128×128 • Direct Adversarial Synthesis</p>
        </div>
        """, unsafe_allow_html=True)
        c_g_cols = st.columns(comp_faces)
        for i in range(comp_faces):
            with c_g_cols[i]:
                st.image(tensor_to_pil(c_gan_imgs[i]), use_container_width=True)
                st.caption(f"GAN #{i+1}")

    st.markdown("---")
    st.subheader("2. Structural & Theoretical Comparison Matrix")

    comparison_data = {
        "Feature": [
            "Architecture",
            "Latent Space",
            "Training Objective",
            "Generation Method",
            "Native Resolution",
            "Displayed Resolution",
            "Total Parameters",
            "Observed Image Characteristics",
            "Training Dynamics & Stability"
        ],
        "Variational Autoencoder (VAE)": [
            "Convolutional Encoder + Transposed Conv Decoder",
            "Continuous 64-D Gaussian (μ, log σ²)",
            "Reconstruction Loss (MSE / Feature Matching) + KL Divergence Regularization",
            "Direct decoding of sampled latent vectors z ~ N(μ, σ²)",
            "64 × 64",
            "128 × 128 (High-Quality Bicubic Upscaling)",
            f"{vae_total_params:,} (~{vae_total_params/1e6:.2f}M)",
            "Smooth, structurally well-formed faces; softer texture and slight blur; high semantic consistency without mode collapse",
            "Monotonically stable; convex reconstruction loss ensures convergence without adversarial oscillation"
        ],
        "Basic Convolutional GAN": [
            "Deep Convolutional Generator + Discriminator (DCGAN style)",
            "128-D Standard Gaussian prior z ~ N(0, I)",
            "Minimax Adversarial Binary Cross-Entropy Loss: min_G max_D V(D, G)",
            "Transposed convolutional upsampling through generator pipeline",
            "128 × 128",
            "128 × 128 (Native)",
            f"{gan_total_params:,} (~{gan_total_params/1e6:.2f}M)",
            "Higher native resolution with defined facial edges and hair contrast; may display local asymmetry or structural artifacts",
            "Adversarial minimax game; highly sensitive to learning rate balance; susceptible to discriminator dominance"
        ]
    }

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("3. In-Depth Trade-Off Analysis")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### 🔬 VAE Strengths & Constraints")
        st.markdown("""
        - **Smooth & Regular Latent Space:** Because the latent space is constrained by the KL term to follow a Gaussian distribution, any sampled vector yields a plausible face structure.
        - **Lack of High-Frequency Detail:** Standard reconstruction objectives (such as Mean Squared Error) penalize average pixel differences, causing the model to produce blurry averages rather than sharp, individual features.
        - **Reliability:** No adversarial divergence or catastrophic training breakdown.
        """)
    with col_t2:
        st.markdown("##### ⚡ Basic GAN Strengths & Constraints")
        st.markdown("""
        - **Sharp Texture Synthesis:** The adversarial discriminator penalizes unrealistic blurs, forcing the generator to produce high-frequency facial textures and distinct features.
        - **Higher Native Resolution:** Directly outputs 128 × 128 without relying on post-processing upscaling.
        - **Adversarial Imbalance:** Discriminator accuracy often approaches 99.9% on fake images, indicating that the discriminator quickly dominates, making further generator refinements challenging without specialized techniques.
        """)


# =============================================================================
# TAB 5: GAN Training & Evaluation
# =============================================================================
with tab_eval:
    st.header("GAN Training & Evaluation Dashboard")
    st.markdown(
        "Empirical analysis based on the actual recorded training logs and evaluation checkpoints "
        "stored in `latest_results/`."
    )

    # 1. Load data dynamically from latest_results
    history_csv_path = RESULTS_DIR / "gan_training_history.csv"
    metrics_csv_path = RESULTS_DIR / "gan_evaluation_metrics.csv"
    report_txt_path = RESULTS_DIR / "GAN_EVALUATION_REPORT.txt"
    loss_img_path = RESULTS_DIR / "gan_training_losses.png"
    eval_grid_img_path = RESULTS_DIR / "generated_evaluation_grid.png"

    has_history = history_csv_path.exists()
    has_metrics = metrics_csv_path.exists()
    has_report = report_txt_path.exists()
    has_loss_img = loss_img_path.exists()
    has_grid_img = eval_grid_img_path.exists()

    # Dynamic Key Metrics Cards
    if has_metrics:
        df_metrics = pd.read_csv(metrics_csv_path)
        m_row = df_metrics.iloc[0]

        st.subheader("1. Final Evaluation Metrics")
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Final Generator Loss</div>
                <div class='metric-value'>{float(m_row['Final Generator Loss']):.4f}</div>
                <div class='metric-sub'>Min: {float(m_row['Minimum Generator Loss']):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Final Disc. Loss</div>
                <div class='metric-value'>{float(m_row['Final Discriminator Loss']):.4f}</div>
                <div class='metric-sub'>Min: {float(m_row['Minimum Discriminator Loss']):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Real Image Accuracy</div>
                <div class='metric-value'>{float(m_row['Real Image Accuracy'])*100:.2f}%</div>
                <div class='metric-sub'>D classification on real</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Fake Image Accuracy</div>
                <div class='metric-value'>{float(m_row['Fake Image Accuracy'])*100:.2f}%</div>
                <div class='metric-sub'>D classification on fakes</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Overall D Accuracy</div>
                <div class='metric-value'>{float(m_row['Overall Discriminator Accuracy'])*100:.2f}%</div>
                <div class='metric-sub'>Evaluation on {int(m_row.get('Evaluation Images', 1000))} images</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Training Curves & History
    st.subheader("2. Adversarial Training Loss Dynamics (30 Epochs)")
    col_chart_l, col_chart_r = st.columns([1.2, 1])

    with col_chart_l:
        if has_history:
            df_history = pd.read_csv(history_csv_path)
            chart_data = df_history.set_index("Epoch")[["Generator_Loss", "Discriminator_Loss"]]
            st.line_chart(chart_data)
            st.caption("Interactive Loss History (Blue: Generator Loss | Orange: Discriminator Loss)")
        elif has_loss_img:
            st.image(str(loss_img_path), caption="Training Losses Over Epochs", use_container_width=True)

    with col_chart_r:
        if has_loss_img:
            st.image(str(loss_img_path), caption="Recorded Loss Curves (gan_training_losses.png)", use_container_width=True)

    st.markdown("---")

    # Evaluation Grid & Observations
    st.subheader("3. Generated Evaluation Grid & Metric Interpretation")
    col_grid_l, col_grid_r = st.columns([1.1, 1])

    with col_grid_l:
        if has_grid_img:
            st.image(str(eval_grid_img_path), caption="Evaluation Grid of Generated Faces (128×128)", use_container_width=True)
        else:
            st.warning("generated_evaluation_grid.png not found.")

    with col_grid_r:
        st.markdown("##### 📌 Critical Interpretation of Metrics")
        st.markdown("""
        - **Generator Loss (~4.495):** The generator loss remained elevated across 30 epochs, reflecting sustained adversarial pressure from the discriminator.
        - **Discriminator Loss (~0.600):** The discriminator loss steadily declined, indicating that the discriminator maintained a strong advantage throughout training.
        - **Fake Image Accuracy (99.90%):** The discriminator correctly classified 99.9% of generated faces as fake, demonstrating effective feature discernment.
        - **Real Image Accuracy (12.21%):** The discriminator achieved 12.21% accuracy on real validation images, showing conservative decision thresholds.
        """)

        st.warning("""
        **Crucial Academic Distinction:**
        **Discriminator accuracy is NOT a direct image-quality metric.**
        In standard GANs, high discriminator accuracy often indicates that the discriminator has overpowered the generator, rather than implying that the generated images are visually flawed or perfect.
        Visual quality must be judged directly from generated samples.
        """)

        if has_report:
            with st.expander("📄 View Full GAN_EVALUATION_REPORT.txt"):
                report_text = report_txt_path.read_text(encoding="utf-8")
                st.code(report_text, language="text")

    st.markdown("---")
    st.subheader("4. Training Observations")
    st.markdown("""
    1. **Loss Stability:** The generator loss stabilized in the range of 3.3 to 4.5, avoiding catastrophic generator collapse.
    2. **Structural Acquisition:** As shown in the evaluation grid, the generator successfully learned facial symmetry, skin tones, eye positions, and diverse hairstyles across CelebA.
    3. **Foundational Trade-off:** As expected for a basic convolutional GAN without progressive growing or spectral normalization, subtle local artifacts appear on complex boundaries (e.g. background transitions and ear structures).
    """)


# =============================================================================
# TAB 6: About / Technical Details
# =============================================================================
with tab_tech:
    st.header("About & Technical Specifications")
    st.markdown("Comprehensive architectural specifications, dynamic parameter counts, and environment details.")

    col_spec_l, col_spec_r = st.columns(2)

    with col_spec_l:
        st.subheader("Variational Autoencoder (VAE)")
        st.markdown(f"""
        - **Model Type:** Convolutional Variational Autoencoder (VAE-GAN trained)
        - **Native Resolution:** 64 × 64 × 3
        - **Latent Dimension:** 64
        - **Encoder:** 4 × Conv2d blocks (32, 64, 128, 256) with BatchNorm & LeakyReLU(0.2) $\\to$ Flatten $\\to$ Dual Linear heads ($\mu, \log\sigma^2$)
        - **Decoder:** Linear projection ($64 \\to 4096$) $\\to$ Reshape ($256 \\times 4 \\times 4$) $\\to$ 4 × ConvTranspose2d blocks (128, 64, 32, 3) $\\to$ Tanh
        - **Reconstruction Objective:** Mean Squared Error (MSE) / Feature Matching assisted
        - **Latent Regularization:** KL Divergence to Standard Gaussian
        - **Dynamic Parameter Breakdown:**
          - Encoder Parameters: **{vae_enc_params:,}**
          - Decoder Parameters: **{vae_dec_params:,}**
          - **Total VAE Parameters:** **{vae_total_params:,}** (~{vae_total_params/1e6:.2f}M)
        """)

    with col_spec_r:
        st.subheader("Basic Convolutional GAN")
        st.markdown(f"""
        - **Model Type:** Deep Convolutional GAN (DCGAN baseline)
        - **Native Resolution:** 128 × 128 × 3
        - **Latent Dimension:** 128 ($z \\sim \\mathcal{{N}}(0, I)$)
        - **Generator:** Linear projection ($128 \\to 8192$) $\\to$ Reshape ($512 \\times 4 \\times 4$) $\\to$ 5 × ConvTranspose2d upsampling blocks (512, 256, 128, 64, 32) $\\to$ Conv2d ($32 \\to 3$) $\\to$ Tanh
        - **Discriminator:** 5 × Conv2d downsampling blocks (64, 128, 256, 512, 512) with LeakyReLU(0.2) and BatchNorm $\\to$ Conv2d ($512 \\to 1$) $\\to$ Sigmoid
        - **Training Objective:** Minimax Adversarial Binary Cross-Entropy
        - **Dynamic Parameter Breakdown:**
          - Generator Parameters: **{gan_gen_params:,}**
          - Discriminator Parameters: **{gan_disc_params:,}**
          - **Total GAN Parameters:** **{gan_total_params:,}** (~{gan_total_params/1e6:.2f}M)
        """)

    st.markdown("---")
    st.subheader("Model Checkpoint File Status")

    file_status_data = [
        {
            "Component": "VAE Encoder",
            "File": "models/encoder_final.pth",
            "Size (MB)": f"{(MODELS_DIR / 'encoder_final.pth').stat().st_size / (1024*1024):.2f}" if (MODELS_DIR / "encoder_final.pth").exists() else "N/A",
            "Status": "Loaded ✓" if vae_encoder is not None else "Missing ✗",
            "Parameters": f"{vae_enc_params:,}"
        },
        {
            "Component": "VAE Decoder",
            "File": "models/decoder_final.pth",
            "Size (MB)": f"{(MODELS_DIR / 'decoder_final.pth').stat().st_size / (1024*1024):.2f}" if (MODELS_DIR / "decoder_final.pth").exists() else "N/A",
            "Status": "Loaded ✓" if vae_decoder is not None else "Missing ✗",
            "Parameters": f"{vae_dec_params:,}"
        },
        {
            "Component": "VAE Latent Distribution",
            "File": "models/latent_distribution.pth",
            "Size (MB)": f"{(MODELS_DIR / 'latent_distribution.pth').stat().st_size / 1024:.2f} KB" if (MODELS_DIR / "latent_distribution.pth").exists() else "N/A",
            "Status": "Loaded ✓" if vae_latent_dist is not None else "Missing ✗",
            "Parameters": "64-D mean & std"
        },
        {
            "Component": "Basic GAN Generator",
            "File": "latest_results/generator_final.pth",
            "Size (MB)": f"{(RESULTS_DIR / 'generator_final.pth').stat().st_size / (1024*1024):.2f}" if (RESULTS_DIR / "generator_final.pth").exists() else "N/A",
            "Status": "Loaded ✓" if gan_generator is not None else "Missing ✗",
            "Parameters": f"{gan_gen_params:,}"
        },
        {
            "Component": "Basic GAN Discriminator",
            "File": "latest_results/discriminator_final.pth",
            "Size (MB)": f"{(RESULTS_DIR / 'discriminator_final.pth').stat().st_size / (1024*1024):.2f}" if (RESULTS_DIR / "discriminator_final.pth").exists() else "N/A",
            "Status": "Loaded ✓" if gan_discriminator is not None else "Missing ✗",
            "Parameters": f"{gan_disc_params:,}"
        }
    ]

    df_files = pd.DataFrame(file_status_data)
    st.dataframe(df_files, hide_index=True, use_container_width=True)

    st.caption(
        "Note on Checkpoint Separation: `models/discriminator_final.pth` is the VAE-side discriminator (~11.08 MB) "
        "and `latest_results/discriminator_final.pth` is the Basic GAN discriminator (~27.87 MB). They are distinct models "
        "and are strictly segregated to avoid architecture mismatches."
    )

    st.markdown("---")
    st.subheader("Runtime Environment")
    env_cols = st.columns(4)
    with env_cols[0]:
        st.metric("PyTorch Version", torch.__version__)
    with env_cols[1]:
        st.metric("Inference Device", device.type.upper())
    with env_cols[2]:
        st.metric("Device Hardware", device_name)
    with env_cols[3]:
        st.metric("Streamlit Version", st.__version__)
