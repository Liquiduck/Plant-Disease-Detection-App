import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import random
import os

with st.sidebar:
    st.header("🌱 About")
    st.markdown("""
    **Plant Disease Detector**  
    Upload a plant leaf photo and see what disease it might have.

    - **Model:** ResNet18, transfer learning
    - **Dataset:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
    - **Validation accuracy:** ~92%
    - **Made by:** Ronas Yüce(https://github.com/Liquiduck)
    """)


# --- Load class names ---
with open("class_names.json") as f:
    class_names = json.load(f)

# --- Load trained model ---
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("plant_disease_resnet18.pth", map_location="cpu"))
model.eval()

# --- Define preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf photo and get an instant disease prediction.")

uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

def predict_topk(img, model, class_names, k=3):
    img = img.convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        topk = torch.topk(probs, k)
        top_classes = [class_names[i] for i in topk.indices.tolist()]
        top_probs = topk.values.tolist()
        return list(zip(top_classes, top_probs))


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Predicting...")

    top_preds = predict_topk(img, model, class_names, k=3)
    
    st.markdown("### Top 3 Predictions")
    for i, (label, prob) in enumerate(top_preds, 1):
        st.write(f"**{i}. {label}** — Confidence: `{prob:.2%}`")

    import pandas as pd
    pred_df = pd.DataFrame(top_preds, columns=['Class', 'Confidence'])
    st.bar_chart(pred_df.set_index('Class'))
else:
    st.info("Please upload a plant leaf image to get started!")
    # Show two example images and their predictions
    st.markdown("#### Example images:")
    sample_folder = "samples"
    example_imgs = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(example_imgs) >= 2:
        cols = st.columns(2)
        for i in range(2):
            img_path = os.path.join(sample_folder, example_imgs[i])
            img = Image.open(img_path)
            cols[i].image(img, use_container_width=True, caption=f"Example {i+1}")
            # Get prediction for the example
            top_preds = predict_topk(img, model, class_names, k=3)
            pred_label, pred_conf = top_preds[0]
            pred_caption = f"**Prediction:** {pred_label}<br>**Confidence:** {pred_conf:.2%}"
            cols[i].markdown(pred_caption, unsafe_allow_html=True)


st.markdown("---")
st.markdown("Made with [Streamlit](https://streamlit.io/) & PyTorch.")
