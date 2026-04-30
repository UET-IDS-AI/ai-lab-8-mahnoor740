'''
AI_stats_lab.py
 
Neural Networks Lab: 3-Layer Forward Pass and Backpropagation
 
Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
'''
 
import numpy as np
 
 
def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))
 
 
def forward_pass(X, W1, W2, W3):
    
   
    # Layer 1: input -> hidden layer 1
    z1 = X @ W1  # Matrix multiplication: input × weights
    h1 = sigmoid(z1)  # Apply sigmoid activation
    
    # Layer 2: hidden layer 1 -> hidden layer 2
    # ⚠️ IMPORTANT: Use h1 here, NOT X
    z2 = h1 @ W2
    h2 = sigmoid(z2)
    
    # Output layer: hidden layer 2 -> output
    z3 = h2 @ W3
    y = sigmoid(z3)
    
    return h1, h2, y
 
 
def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    
    
    n_samples = X.shape[0]
    
    # Reshape label if it's a scalar
    if np.isscalar(label):
        label = np.array([[label]])
    elif label.ndim == 1:
        label = label.reshape(-1, 1)
    
    # Step 1: Calculate loss (Binary Cross-Entropy)
    # This measures how far our prediction is from the true answer
    # Loss = -[label*log(y) + (1-label)*log(1-y)]
    loss = -np.mean(label * np.log(y + 1e-8) + (1 - label) * np.log(1 - y + 1e-8))
    
    # Step 2: Output layer error
    # How much did we miss? (prediction - true answer)
    dz3 = y - label  # This is the error at output
    
    # Step 3: Calculate gradient for W3
    # dW3 = h2.T @ dz3 / n_samples
    # We use h2 because W3 connects h2 to output
    dW3 = (h2.T @ dz3) / n_samples
    
    # Step 4: Backpropagate error to h2
    # Multiply error by W3 and apply sigmoid derivative
    dh2 = dz3 @ W3.T  # Pass error backwards through W3
    dz2 = dh2 * h2 * (1 - h2)  # Apply sigmoid derivative
    
    # Step 5: Calculate gradient for W2
    # dW2 = h1.T @ dz2 / n_samples
    dW2 = (h1.T @ dz2) / n_samples
    
    # Step 6: Backpropagate error to h1
    # Multiply error by W2 and apply sigmoid derivative
    dh1 = dz2 @ W2.T
    dz1 = dh1 * h1 * (1 - h1)
    
    # Step 7: Calculate gradient for W1
    # dW1 = X.T @ dz1 / n_samples
    dW1 = (X.T @ dz1) / n_samples
    
    return dW1, dW2, dW3, loss
