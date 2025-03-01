# 🌱 Plant Disease Detection using CNN 

 A CNN-based plant disease detection project using TensorFlow.

 ## 🔗 Download the Model
The trained model is available in [GitHub Releases](https://drive.google.com/file/d/1rAlAPDuYQaJY5LhiR4Rk2vaxReY-Zx3W/view?usp=drive_link)

datasets (https://drive.google.com/drive/folders/1yGPbeCcsLzMS4yd8_VO65Eua9awTJoHD?usp=drive_link)


# 🌱 Plant Disease Detection using CNN  

This project uses **Convolutional Neural Networks (CNN)** to detect plant diseases from images.  
It is built using **TensorFlow, Keras, and Google Colab**, and trained on a dataset stored in **Google Drive**.

---

## 📷 Sample Output  
Below is an example of the model's prediction on a test image:  

![Test Image Prediction](https://drive.google.com/file/d/1SitU8Q0UIMcdamvECQ00LAevLLzpnuyr/view?usp=drive_link)  

---

## 🚀 Features
✅ Deep Learning-based **plant disease classification**  
✅ **Google Colab-compatible** (no local setup required)  
✅ **Uses TensorFlow and Keras** for model training  
✅ **Pretrained model available** for direct inference  
✅ **Dataset stored in Google Drive** for easy access  

---

## 📂 Dataset Information  
The dataset is stored in **Google Drive** and contains images of different plant diseases.  

### **🔹 Dataset Structure**
/datasets_plant/ ├── train/ (Training images) │ ├── Blight/ │ ├── Common Rust/ │ ├── / │ ├── Grey Leaf Spot / |├── Healthy / | 

## 🏗 Model Architecture  
This project uses a **CNN-based deep learning model** with the following layers:

1️⃣ **Conv2D + ReLU Activation**  
2️⃣ **MaxPooling2D**  
3️⃣ **Conv2D + ReLU Activation**  
4️⃣ **MaxPooling2D**  
5️⃣ **Flatten Layer**  
6️⃣ **Dense (128 units, ReLU)**  
7️⃣ **Dense (Softmax for classification)**  

---

## 🖥 How to Run This Project  

### **1️⃣ Open Google Colab & Mount Drive**
1. Open [Google Colab](https://colab.research.google.com/).
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
2️⃣ Load the Dataset
Ensure the dataset is inside your Google Drive at:
/content/drive/MyDrive/datasets_plant/train
train_dir = "/content/drive/MyDrive/datasets_plant/train"

3️⃣ Train the Model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
Once training is complete, the model will be saved automatically.

4️⃣ Download or Use Pretrained Model
Since GitHub does not support large files, the trained model (.h5 file) is stored separately.

📥 Download Pretrained Model from Google Drive.
After downloading, place it in the working directory and load it using:

from tensorflow.keras.models import load_model
model = load_model('plants_diseases_cnn_v1.h5')

🎯 Prediction on New Images
Use the trained model to predict plant diseases from new images:


import numpy as np
from tensorflow.keras.preprocessing import image
test_image_path = "path_to_your_test_image.jpg"
img = image.load_img(test_image_path, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Predicted: {list(class_indices.keys())[predicted_class]} | Confidence: {confidence:.2f}%")
📌 Technologies Used
🔹 Python
🔹 TensorFlow & Keras
🔹 Google Colab
🔹 Matplotlib (for visualization)
🔹 NumPy & OpenCV (for image processing)

📜 License
This project is open-source under the MIT License.

💡 Author
👨‍💻 K Uma Maheshwari

📧 Email: keerthanakrishanamoorthy@gmail.com

🔗 GitHub: https://github.com/keerthana824

⭐ Star This Repository!
If you found this project useful, consider starring 🌟 the repository on GitHub!

---

## **🎯 What This README Includes**
✔️ **Project Description**  
✔️ **Dataset Structure**  
✔️ **Step-by-step Execution Guide**  
✔️ **Pretrained Model Download**  
✔️ **Prediction Code**  
✔️ **Tech Stack & Author Info**  

Let me know if you need any modifications...!!




















