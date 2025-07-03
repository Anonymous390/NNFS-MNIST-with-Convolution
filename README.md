![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-%E2%9C%94-lightgrey)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![MNIST](https://img.shields.io/badge/dataset-MNIST-orange)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Anonymous390/YourMNISTRepo)
<h1 align="center">CNN from Scratch on MNIST</h1>

This project implements a simple Convolutional Neural Network (CNN) using **NumPy only**, without any deep learning frameworks like TensorFlow or PyTorch.  
It trains on the **MNIST handwritten digit dataset** (28×28 grayscale images) and includes forward and backward propagation, convolution, pooling, dense layers, softmax + cross-entropy loss, and SGD optimizer with momentum + decay.  

Achieves ~98% accuracy on MNIST test set after 5 epochs.

---

## 🚀 Features  
✅ Convolutional layer (`Conv2D`) with Xavier/He initialization  
✅ Max Pooling layer (`MaxPool2D`)  
✅ Fully-connected (dense) layers  
✅ ReLU and Softmax activations  
✅ Categorical cross-entropy loss  
✅ SGD optimizer with momentum and learning rate decay  
✅ Custom forward + backward pass  
✅ Model saving and loading  
✅ Support for custom image predictions  

---

## 🗂 Dataset  
- MNIST digits (via `keras.datasets.mnist`)  
- 60,000 train / 10,000 test samples  
- Images normalized to `[0,1]`  

---

## 🧱 Model Architecture  
```
Input: 28x28x1  

Conv2D: 3x3 kernel, 32 filters, padding=1  
→ ReLU  
→ MaxPool2D: 2x2  

Flatten  

Dense: 128 units  
→ ReLU  

Dense: 10 units  
→ Softmax
```

---

## 📈 Training  
- Optimizer: SGD with momentum = 0.9, decay = 1e-3  
- Batch size: 64  
- Epochs: 5  

Example output:
```
Epoch 1: Loss = 0.5501, Accuracy = 0.8502  
Epoch 2: Loss = 0.4003, Accuracy = 0.8904  
...
```

---

## 💻 Running the code  
```bash
pip install keras nnfs matplotlib opencv-python pillow
```

👉 **Run the notebook or script:**  
```python
# Example forward + training
loss = forward_pass(X_batch, y_batch)
backward_pass(loss_func.output, y_batch)
optimizer.pre_update_params()
optimizer.update_params(layer2)
optimizer.update_params(layer1)
optimizer.update_params(conv1)
optimizer.post_update_params()
```

---

## 🔍 Custom image prediction  
You can load your own image:
```python
test_image = cv2.imread("pred1.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28,28))
test_image = cv2.bitwise_not(test_image) / 255.0
```
And predict:
```python
conv1.forward(img)
...
print("Predicted digit:", predicted_class[0])
```
Example output:
```
0: 0.000001 %
1: 0.000002 %
...
8: 99.998234 %
9: 0.000100 %
Predicted digit: 8
```

---

## 💾 Saving & Loading model  
```python
# Save model
saver = ModelSaver()
saver.save_model(conv1, layer1, layer2, optimizer)

# Load model
saver.load_model(conv1, layer1, layer2, optimizer)
```
Weights, biases, optimizer state saved in `.npz` file.

---

## 📊 Visualizing wrong predictions  
Plots incorrect guesses with predicted + true labels:
```python
for idx in first_10_incorrect:
    plt.imshow(X_test[idx], cmap="gray")
    print(f"Predicted: {test_predictions[idx]}, True: {test_true[idx]}")
```

---

## 📌 File structure  
```
├── cnn_mnist.py / notebook.ipynb
├── model.npz         # Saved model
├── pred1.png         # Example custom test image
└── README.md
```

---

## 🙌 Credits  
Inspired by [NNFS book](https://nnfs.io/) and built using **NumPy**, **OpenCV**, **Matplotlib**.

---

## ⭐️ Future work  
- Add more conv layers  
- Add dropout / batchnorm  
- Add support for other datasets (e.g. CIFAR-10)
