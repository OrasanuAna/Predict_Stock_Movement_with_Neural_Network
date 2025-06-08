# 🧠 Predict Stock Movement with Neural Network

This project demonstrates how to build a simple neural network to predict **stock movement direction** (up or down) based on historical price data. It uses **TensorFlow/Keras** for model implementation and focuses on binary classification.

## 📌 Objective

Predict whether the stock price will go **up** or **down** the next day using features such as Open, High, Low, Close, and Volume.

## 🚀 Features

* Load and preprocess historical stock data
* Convert price changes into binary labels (up/down)
* Build and train a neural network using Keras
* Evaluate the model accuracy
* Visualize predicted vs actual movements

## 🛠️ Technologies Used

* Python
* pandas
* NumPy
* TensorFlow / Keras
* matplotlib

## 📂 Project Structure

```
Predict_Stock_Movement_with_Neural_Network/
├── main.py                                      # Main script for training and prediction
├── stock_data.csv                               # Input data file
├── Problema5 - Predict Stock Movement with Neural Network.jpg  # Output image
└── README.md
```

## 📊 Sample Output

![Prediction Output](https://github.com/OrasanuAna/Predict_Stock_Movement_with_Neural_Network/blob/master/Problema5%20-%20Predict%20Stock%20Movement%20with%20Neural%20Network.jpg)

*Figure: Comparison between actual and predicted stock movements (Up = 1, Down = 0).*

## 🧠 Model Architecture

* Input Layer: Features from historical stock data
* Hidden Layers: 1–2 dense layers with ReLU activation
* Output Layer: Sigmoid activation for binary classification

## 📈 How to Run

1. **Clone the Repo**

   ```bash
   git clone https://github.com/OrasanuAna/Predict_Stock_Movement_with_Neural_Network.git
   cd Predict_Stock_Movement_with_Neural_Network
   ```

2. **Install Dependencies**

   ```bash
   pip install pandas numpy matplotlib tensorflow
   ```

3. **Run the Script**

   ```bash
   python main.py
   ```

4. **Interpret Results**

   * Accuracy and performance metrics will be printed in the console.
   * A plot will show the predicted vs actual movements.

