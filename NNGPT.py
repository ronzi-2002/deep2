import numpy as np

class ActivationFunction:
    def __init__(self, name):
        self.name = name
    
    def apply(self, x):
        if self.name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return np.maximum(0, x)
        elif self.name == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Activation function not supported")

    def derivative(self, x):
        if self.name == 'sigmoid':
            return x * (1 - x)
        elif self.name == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.name == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError("Activation function not supported")

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid'):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5
        self.activation = ActivationFunction(activation)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation.apply(np.dot(inputs, self.weights) + self.biases)
        return self.output

    def backward(self, dvalues, learning_rate):
        # Compute gradients
        dinputs = np.dot(dvalues, self.weights.T)
        dweights = np.dot(self.inputs.T, dvalues)
        dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases
        
        return dinputs

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dvalues, learning_rate):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues, learning_rate)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for input_data, target_data in zip(X, y):
                # Forward pass
                output = self.forward(input_data.reshape(1, -1))

                # Calculate loss
                loss = np.mean((output - target_data) ** 2)

                # Backward pass
                dvalues = 2 * (output - target_data)
                self.backward(dvalues, learning_rate)

            # Print loss for monitoring training progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Example usage:
# Define a neural network
nn = NeuralNetwork()
nn.add_layer(Layer(input_size=2, output_size=5, activation='tanh'))
nn.add_layer(Layer(input_size=5, output_size=5, activation='tanh'))
nn.add_layer(Layer(input_size=5, output_size=5, activation='tanh'))
nn.add_layer(Layer(input_size=5, output_size=2, activation='tanh'))
nn.add_layer(Layer(input_size=2, output_size=1, activation='sigmoid'))

# Training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([1, 0, 0, 1]).reshape(-1, 1)  # Example targets for XOR

# Train the neural network
nn.train(inputs, targets, learning_rate=0.1, epochs=100)

# Test the neural network
for input_data in inputs:
    print(f"Input: {input_data}, Predicted: {nn.forward(input_data.reshape(1, -1))}")