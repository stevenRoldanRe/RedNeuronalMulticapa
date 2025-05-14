import numpy as np

# Funciones de activación y derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Entrenamiento (X: entradas, y: salidas esperadas)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])  # XOR como ejemplo

# Inicialización de pesos y sesgos (biases)
np.random.seed(42)
input_size = X.shape[1]
hidden_size = 3
output_size = 1

# Pesos
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Entrenamiento
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # salida predicha

    # Cálculo del error
    error = y - a2

    # Backpropagation
    d_a2 = error * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    # Actualización de pesos y sesgos
    W2 += learning_rate * d_W2
    b2 += learning_rate * d_b2
    W1 += learning_rate * d_W1
    b1 += learning_rate * d_b1

    # Mostrar error cada 1000 iteraciones
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# Predicciones finales
print("\nPredicciones:")
print(a2)
