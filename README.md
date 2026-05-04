# 🐾 Animalia: 90-Class Deep Learning Classifier
### **Bridging Computer Vision & Real-World Accessibility**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)]https://huggingface.co/spaces/mubashrawaqar123/AnimalClassifier
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Animalia** is a specialized image recognition engine capable of identifying **90 different animal species** with high precision. This project demonstrates the power of **Transfer Learning**, taking complex, research-grade architectures like **ResNet-50** and deploying them into a functional, user-centric web application.

---

##  [Try the Live Demo Here] https://huggingface.co/spaces/mubashrawaqar123/AnimalClassifier


##  The "Why" Behind the Project
I wanted to build a complete pipeline, from raw data cleaning to a live interface. 

The core challenge was the scale of the classes. Distinguishing between subtle species (like a **Coyote** vs. a **Wolf**) required more than basic CNN layers; it required deep residual learning to capture the fine-grained textures and anatomical markers of the animal kingdom.

## Technical Stack & Architecture
* **Deep Learning Framework:** `PyTorch`
* **Model Architecture:** `ResNet-50` (Residual Network)
* **Optimization:** `Adam Optimizer` with Cross-Entropy Loss.
* **Web Interface:** `Streamlit`
* **Deployment:** `Hugging Face Spaces`

### **Key Technical Decisions:**
* **Transfer Learning:** I leveraged pre-trained weights from ImageNet-1K to give the model a "head start" on basic shapes and edges, then fine-tuned the final layers for my 90 specific classes.
* **Preprocessing:** All input images are normalized and resized to $224 \times 224$ pixels to match the ResNet input layer expectations.
* **Inference Strategy:** Since the app is hosted on a CPU-based environment, the model uses a cached state-dictionary to ensure quick predictions without needing a GPU.

##  Performance Metrics
The model was trained for 5 epochs with a high learning rate of $0.001$, achieving excellent convergence:

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | **96.18%** |
| **Validation Accuracy** | **91.11%** |

## Project Structure
* `app.py` - The logic for the Streamlit UI and PyTorch inference.
* `requirements.txt` - Dependency list for the cloud environment.
* `animal_resnet50_final.pth` - *Hosted on Hugging Face (File size >100MB)*.

## 🧠 Reflections & Learning
The biggest hurdle wasn't the code—it was the data. Real world datasets are messy. Handling hidden system files, cleaning empty directories, and optimizing memory usage for a 100MB model on a free-tier server taught me more about **Software Engineering** than a textbook ever could.

---
*Created with 🐾 as part of my BSCS Semester Project.*
