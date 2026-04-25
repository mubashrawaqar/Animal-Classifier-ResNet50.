import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# THIS MUST BE THE FIRST ST COMMAND
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")

# 2. NOW YOU CAN DO OTHER ST COMMANDS
st.title("🐾 90-Species Animal Classifier")
st.write("Upload an image, and the ResNet-50 model will identify the animal.")

# 1. THE LIST (Exactly as your model learned them)
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

# 2. LOAD MODEL ARCHITECTURE
@st.cache_resource
def load_model():
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 90)
    # Load weights - we use map_location=cpu because HF Free Tier doesn't have GPU
    model.load_state_dict(torch.load("animal_resnet50_final.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. STREAMLIT UI


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    st.success(f"Prediction: **{class_names[predicted.item()].upper()}**")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")
