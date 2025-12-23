# CodeAlpha_Handwritten_Character_Recognition
Handwritten Character Recognition using CNN (MNIST Dataset) built as part of the CodeAlpha Machine Learning Internship.

# CodeAlpha_Handwritten_Character_Recognition

Handwritten Character Recognition system built using **Deep Learning (CNN)**  
as part of the **CodeAlpha Machine Learning Internship**.

This project classifies handwritten digits (0–9) using the **MNIST dataset**.


## 📌 Project Objective
- Recognize handwritten digits from grayscale images  
- Apply Convolutional Neural Networks (CNN)  
- Evaluate model performance using accuracy and loss metrics  

## 📊 Dataset
- **Dataset Name:** MNIST Handwritten Digits  
- **Training Samples:** 60,000  
- **Testing Samples:** 10,000  
- **Image Size:** 28 × 28 (grayscale)  
- **Classes:** Digits (0–9)


## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Google Colab  

## 🧠 Model Architecture
- Input Layer (28×28×1)
- Convolutional Layer (ReLU)
- Max Pooling
- Convolutional Layer (ReLU)
- Max Pooling
- Fully Connected Dense Layer
- Dropout (Regularization)
- Output Layer (Softmax – 10 classes)

## 🚀 Model Training
- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy  
- Epochs: 10  
- Batch Size: 128  

## 📈 Results
- **Training Accuracy:** ~99%  
- **Validation Accuracy:** ~99%  
- **Test Accuracy:** ~98–99%  

The model shows excellent performance and generalization ability.

## 📊 Visualizations
- Training vs Validation Accuracy Graph  
- Training vs Validation Loss Graph  
- Sample handwritten digit visualization  
- Confusion Matrix  


## 📁 Project Structure


CodeAlpha_Handwritten_Character_Recognition/
│
├── Handwritten_Character_Recognition.ipynb
├── model.keras
├── README.md


## ✅ Conclusion
The CNN model successfully learned spatial features from handwritten digits
and achieved high accuracy on unseen data.  
This project demonstrates the effectiveness of deep learning in image
classification tasks and can be extended to recognize characters, words,
or sentences in the future.

## 👩‍💻 Author
**Aneela**  
IT Student | Aspiring AI Engineer  
