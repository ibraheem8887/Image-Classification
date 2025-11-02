import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

st.set_page_config(page_title="🍽️ Food Classification (ViT)", layout="centered")
st.title("🍽️ Food Image Classification (ViT)")
st.write("Upload a food image to classify it into one of the Food-41 categories.")

@st.cache_resource
def load_model():
    weights_path = "vit_food41/vit_food41_full_model.pth"

    if not os.path.exists(weights_path):
        st.error(f"❌ Model file not found at `{weights_path}`")
        st.stop()

    st.info("🔍 Loading Vision Transformer model...")

    # 👇 SAFE for your own trained file
    model = torch.load(weights_path, map_location="cpu", weights_only=False)

    # If model was saved using model.save_pretrained, handle differently:
    if isinstance(model, dict):  # weights only
        vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=41
        )
        vit.load_state_dict(model)
        model = vit

    model.eval()
    st.success("✅ Model loaded successfully!")
    return model

model = load_model()

# ---- CLASS NAMES ----
CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries"
]

# ---- IMAGE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        if hasattr(outputs, "logits"):  # HuggingFace-style model
            outputs = outputs.logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    st.markdown(f"### 🍛 Predicted Class: **{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence: `{confidence:.2f}%`")

else:
    st.info("⬆️ Please upload an image to start classification.")
