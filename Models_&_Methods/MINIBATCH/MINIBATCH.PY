import numpy as np


#X is a dataset of 1000 random samples with one feature.
#y is the target value generated using the formula 𝑦=2𝑋+3, with some random noiseadded to simulate real-world data. This noise ensures the data is not perfectly linear.
X = np.random.randn(1000, 1)  # 1000 samples, 1 feature
y = 2 * X + 3 + np.random.randn(1000, 1)  # Linear relationship with some noise


#Learning rate: Controls the step size of the parameter updates. A smaller value ensures slow but steady convergence.
#Batch size: Each mini-batch consists of 32 samples, which balances computation and convergence stability.
#Epochs: The number of times the model iterates through the entire dataset.
learning_rate = 0.01
batch_size = 32
epochs = 1000


#w (weight) and b (bias) are initialized randomly. These are the parameters the algorithm will learn to minimize the cost function.
w = np.random.randn(1)
b = np.random.randn(1)


#Define Cost Function
def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X.dot(w.T) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


#Mini-batch Gradient Descent
indices = np.random.permutation(X.shape[0])
X_shuffled = X[indices]
y_shuffled = y[indices]
#Shuffling ensures that mini-batches are not biased by the original order of the data.

#Process Each Mini-batch
for i in range(0, X.shape[0], batch_size):
    X_batch = X_shuffled[i:i + batch_size]
    y_batch = y_shuffled[i:i + batch_size]
#The dataset is divided into smaller subsets (mini-batches) of size 32. For example:
#Batch 1: Samples 0–31
#Batch 2: Samples 32–63
#And so on...


#Forward Propagation
predictions = X_batch.dot(w.T) + b
#The model predicts 
#y using the formula ℎ(𝑥𝑖)=𝑤⋅𝑥𝑖+𝑏.


# Compute Gradients
dw = (1 / m) * np.sum((predictions - y_batch) * X_batch)
db = (1 / m) * np.sum(predictions - y_batch)

#Update Parameters
w -= learning_rate * dw
b -= learning_rate * db

#Monitor Training
cost = compute_cost(X, y, w, b)
if epoch % 100 == 0:
    print(f"Epoch {epoch}, Cost: {cost}")
    #After every 100 epochs, the cost is calculated for the entire dataset to track the model's performance.


# Key Concepts in the Code

# Mini-batching:
# - Reduces computation time compared to full-batch gradient descent.
# - Provides stability compared to stochastic gradient descent (SGD).
# - In mini-batch gradient descent, we process small batches of data at a time, which helps to speed up the training process and avoid the computational cost of processing the entire dataset at once.

# Shuffling:
# - Ensures that mini-batches are random, which helps to reduce bias caused by the original order of the data.
# - By shuffling the dataset, we ensure that the model sees a diverse set of data in each mini-batch, leading to better generalization.

# Learning Rate:
# - The learning rate controls the step size for parameter updates during training.
# - A smaller learning rate ensures the model converges slowly, avoiding overshooting the optimal minimum. However, it may require more iterations to converge.
# - A larger learning rate speeds up convergence but may risk overshooting the minimum and failing to converge.

# Gradients:
# - Gradients guide the updates for weight (w) and bias (b) during the optimization process.
# - The gradients are computed based on the partial derivatives of the cost function.
# - Gradients tell us how to adjust the parameters to minimize the cost function and improve the model's predictions.