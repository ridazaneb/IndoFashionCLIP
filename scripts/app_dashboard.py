# scripts/app_dashboard.py

import os
import pickle

import streamlit as st
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ───────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────
BASE_DIR   = '/content/drive/MyDrive/IndoFashion'
MODEL_DIR  = os.path.join(BASE_DIR, 'models/clip_south_asia')
TEST_CSV   = os.path.join(BASE_DIR, 'data/test.csv')
CLUSTER_CSV= os.path.join(BASE_DIR, 'data/train_with_clusters.csv')
CLUSTER_PKL= os.path.join(BASE_DIR, 'models/pca_km.pkl')

# Try local SSD first, else fall back to Drive
IMG_ROOT = '/content/images'
if not os.path.isdir(IMG_ROOT):
    IMG_ROOT = os.path.join(BASE_DIR, 'images')

# ───────────────────────────────────────────────────────────
# RESOURCE LOADING
# ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = CLIPModel.from_pretrained(MODEL_DIR).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)
    classes   = pd.read_csv(TEST_CSV)['class_label'].unique().tolist()
    return model, processor, device, classes

@st.cache_resource(show_spinner=False)
def load_cluster_data():
    df      = pd.read_csv(CLUSTER_CSV)
    with open(CLUSTER_PKL, 'rb') as f:
        pca_km = pickle.load(f)
    return df, pca_km['pca'], pca_km['km']

model, processor, device, class_list = load_model_and_classes()

# Cluster artifacts may not exist yet
try:
    df_clusters, pca, km = load_cluster_data()
    clustering_enabled = True
except Exception:
    clustering_enabled = False

# ───────────────────────────────────────────────────────────
# APP UI
# ───────────────────────────────────────────────────────────
st.set_page_config(page_title="South Asian Fashion Classifier")
st.title("🌺 South Asian Fashion Classifier")

uploaded = st.file_uploader("Upload an image of a garment", type=['jpg','png','jpeg'])
if uploaded:
    # Display upload
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Your Upload", use_container_width=True)

    # Preprocess & forward pass
    inputs = processor(images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        text_inputs = processor(
            text=[f"a {c}" for c in class_list],
            return_tensors="pt",
            padding=True
        ).to(device)

        txt_feats = model.get_text_features(**text_inputs)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims  = (img_feats @ txt_feats.T)[0]
        probs = sims.softmax(dim=0)
        top3  = torch.topk(probs, k=3)

    # Show Top-3 predictions
    st.subheader("Top 3 Predictions")
    for score, idx in zip(top3.values, top3.indices):
        cls = class_list[idx]
        st.write(f"• **{cls.upper()}** — {float(score):.2%}")

    # Optional: Similar styles gallery via clustering
    if clustering_enabled:
        emb   = img_feats.cpu().numpy()
        emb_p = pca.transform(emb)
        cid   = int(km.predict(emb_p)[0])

        st.markdown(f"### Similar styles (cluster {cid})")
        cols = st.columns(4)
        samples = df_clusters[df_clusters.cluster == cid] \
                      .sample(4, random_state=42)['image_path'].tolist()

        for col, img_path in zip(cols, samples):
            full_path = os.path.join(IMG_ROOT, img_path)
            try:
                thumb = Image.open(full_path).convert('RGB')
                col.image(thumb, use_container_width=True, caption=os.path.basename(img_path))
            except Exception:
                col.write("❓ missing image")

else:
    st.info("Please upload a JPG/PNG image of a South Asian garment to get started.")

