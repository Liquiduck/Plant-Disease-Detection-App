import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json


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

def predict(img):
    img = img.convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        return class_names[pred.item()], conf.item()

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Predicting...")
    label, confidence = predict(img)
    st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
else:
    st.info("Please upload a plant leaf image to get started!")

st.markdown("---")
st.markdown("Made with [Streamlit](https://streamlit.io/) & PyTorch.")
