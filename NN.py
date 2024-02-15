#implementing a simple neural network
import numpy as np
class NN:
    def __init__(self, dims):
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(Layer(dims[i-1], dims[i]))#initializing the layers.
            if i != len(dims) - 1:
                self.layers.append(Activation("tanh"))
            else :
                self.layers.append(Activation("softmax"))
        self.lastForwardVals = []

    def forward(self, x):
        self.lastForwardVals = []
        for layer in self.layers:
            x = layer.forward(x)
            self.lastForwardVals.append(x)
        return x
    def backward(self, x, y, y_pred):
        print("lastForwardVals" + str(self.lastForwardVals))
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].__class__.__name__ == "Layer":
                y_pred = self.layers[i].backward(self.lastForwardVals[i-1], y, y_pred)
            else:
                y_pred = y_pred * self.layers[i].backward(self.lastForwardVals[i-1], y, y_pred)
    def train(self, X, y, num_epochs):
        for i in range(num_epochs):
            y_pred = self.forward(X)
            # print("y_pred" + str(y_pred))
            self.backward(self.lastForwardVals[-3], y, y_pred)
            
    def predict(self, X):
        return self.forward(X)
    
    def __str__(self):
        for layer in self.layers:
            print(layer)

class Layer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.rand(input_dim, output_dim)
        self.b = np.random.rand(output_dim)
        self.lr = 0.1

    def forward(self, x):
        # print("W: " + str(self.W) + " b: " + str(self.b))
        # print("WShape: " + str(self.W.shape) + " bShape: " + str(self.b.shape) + " xShape: " + str(x.shape))
        return np.dot(x, self.W) + self.b

    def backward(self, x, y, y_pred):
        print("x" + str(x))
        print("y" + str(y))
        print("x.Shape" + str(x.shape))
        print("y.Shape" + str(y.shape))
        grad_w = np.dot(x.T, y_pred - y)/x.shape[0]
        print("grad_w" + str(grad_w))
        grad_b = np.sum(y_pred - y, axis=0)/x.shape[0]
        self.W -= self.lr * grad_w
        self.b -= self.lr * grad_b
        return np.dot(y_pred - y, self.W.T)
    def __str__(self):
        return "W: " + str(self.W) + " b: " + str(self.b)
class Activation:
    def __init__(self, activation):
        self.activation = activation

    def forward(self, x):
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "softmax":
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, x,y,y_pred):
        if self.activation == "relu":
            return x > 0
        elif self.activation == "sigmoid":
            return x * (1 - x)
        elif self.activation == "tanh":
            return 1 - x**2
        elif self.activation == "softmax":
            return x * (1 - x)
    def __str__(self):
        return str(self.activation)
        
if __name__ == "__main__":
    X = np.random.rand(1, 2)
    y = np.random.rand(1, 2)
    nn = NN([2, 3, 2])
    # print(nn.__str__()) 
    nn.train(X, y, 100)
    print(nn.predict(X))
    print(y)
    print(nn.predict(X) - y)
    print(np.sum(nn.predict(X) - y))