import streamlit as st
import torch
import cv2
import numpy as np
from model import UNet

st.sidebar.selectbox("Pages", ["Dashboard", "Prediction"])
st.set_page_config(layout="wide")

model = UNet(23)
model.load_state_dict(torch.load("unet_city.pth", map_location="cpu"))
model.eval()

page = st.sidebar.selectbox("Select Page", ["Dashboard", "Prediction"])

# ---------------- PAGE 1 ----------------
if page == "Dashboard":
    st.title("Training Results")

    st.image("loss.png")
    st.image("iou.png")
    st.image("dice.png")

    st.write("Test mIoU: XX.XX")
    st.write("Test mDice: XX.XX")


# ---------------- PAGE 2 ----------------
if page == "Prediction":
    st.title("Segmentation Demo")

    files = st.file_uploader("Upload 4 images", accept_multiple_files=True)

    for file in files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96))

        inp = torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float()

        pred = model(inp)
        mask = torch.argmax(pred, dim=1).squeeze().numpy()

        st.image(img, caption="Input")
        st.image(mask, caption="Predicted Mask")
