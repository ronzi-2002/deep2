#implementing a simple neural network
from Qs2.deep import cross_entropy_loss_batch
import numpy as np
class NN:
    def __init__(self, dims, lr=0.1):
        self.layers = []
        # for i in range(1, len(dims)):
        #     self.layers.append(Layer(dims[i-1], dims[i]))#initializing the layers.
        #     if i != len(dims) - 1:
        #         self.layers.append(Activation("tanh"))
        #     else :
        #         self.layers.append(Activation("softmax"))
        # self.lastForwardVals = []
        for i in range(1, len(dims)):
            if i != len(dims) - 1:
                self.layers.append(Layer(dims[i-1], dims[i], Activation("tanh"), lr=lr))
            else:
                self.layers.append(Layer(dims[i-1], dims[i], Activation("softmax"), lr=lr))
        self.loss= None

    def forward(self, x):
        self.lastForwardVals = []
        final_val = None
        for layer in self.layers:
            x = layer.forward(x)
            self.lastForwardVals.append(x)
            final_val= x
        # print("last val: " + str(final_val) + " y: " + str(y))
        loss = cross_entropy_loss_batch(y.T, final_val)
        return x, loss
    
    def backward(self):
        #we treat the last layer differently
        dx,dw,db = self.get_last_dx_dw_db()
        self.layers[-1].W -= self.layers[-1].lr * dw
        self.layers[-1].b -= self.layers[-1].lr * db
        for layer in reversed(self.layers[:-1]):
            dx = layer.update_weights(dx)
        
    def get_last_dx_dw_db(self):
        lastLayer = self.layers[-1]
        # print("shape: " + str(lastLayer.lastForwardValAfterActivation.shape) + " - " + str(y.shape))
        # print("shape: " + str(lastLayer.W.shape))   
        dx=  np.mean(np.dot(lastLayer.W.T,(lastLayer.lastForwardValAfterActivation.T - y).T))
        grad_w = np.dot( lastLayer.lastInput, lastLayer.lastForwardValAfterActivation.T - y)/lastLayer.lastInput.shape[0]
        #if minimal value of y_pred -y is positive, print it
        # if np.min(y_pred - y) > 0:
        # print("y_pred - y" + str(np.min(y_pred - y))+str(np.max(y_pred - y)))
        grad_b = np.sum( lastLayer.lastForwardValAfterActivation.T - y, axis=0)/lastLayer.lastInput.shape[0]
        return dx, grad_w.T, grad_b.reshape(2,1)
        return
        return np.dot((lastLayer.lastForwardValAfterActivation - y), lastLayer.W.T)
    def train(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            # print("weights: " + str(self.layers[0].W) + " biases: " + str(self.layers[0].b))
            y_pred, loss = self.forward(X)
            self.loss = loss
            self.backward()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss))
    def predict(self, X):
        return self.forward(X)
    
    def __str__(self):
        for layer in self.layers:
            print(layer)

class Layer:
    def __init__(self, input_dim, output_dim,activation, lr=0.1):
        # self.W = np.random.rand(input_dim, output_dim)
        self.W = np.random.rand(output_dim, input_dim)
        self.b = np.random.rand(output_dim,1)
        self.lr = lr
        self.activation = activation
        self.lastForwardValAfterActivation = None
        self.lastForwardValBeforeActivation = None
        self.lastInput = None

    def forward(self, x):
        # print("W: " + str(self.W) + " b: " + str(self.b))
        # print("WShape: " + str(self.W.shape) + " bShape: " + str(self.b.shape) + " xShape: " + str(x.shape))
        self.lastInput = x

        self.lastForwardValBeforeActivation = np.dot(self.W, x) + self.b
        self.lastForwardValAfterActivation = self.activation.forward(self.lastForwardValBeforeActivation)
        return self.lastForwardValAfterActivation

    def gradient(self, dx_from_next_layer):
        # print("dx_from_next_layer: " + str(dx_from_next_layer))
        # print("self.activation.backward(self.lastForwardValBeforeActivation): " + str(self.activation.backward(self.lastForwardValBeforeActivation)))
        # print("self.lastInput.T: " + str(self.lastInput.T))
        grad_w = dx_from_next_layer * np.dot(self.activation.backward(self.lastForwardValBeforeActivation), self.lastInput.T)
        grad_b = np.sum(dx_from_next_layer * self.activation.backward(self.lastForwardValBeforeActivation), axis=1, keepdims=True)
        current_dx = np.dot(self.W.T,(dx_from_next_layer * self.activation.backward(self.lastForwardValBeforeActivation)))
        return grad_w, grad_b, current_dx
        
    def update_weights(self, dx_from_next_layer):
        grad_w, grad_b, current_dx = self.gradient(dx_from_next_layer)
        # print("grad_w: " + str(grad_w))
        # print("grad_b: " + str(grad_b))
        self.W -= self.lr * grad_w
        self.b -= self.lr * grad_b
        return current_dx
    

    # def backward(self, x, y, y_pred):
    #     print("x" + str(x))
    #     print("y" + str(y))
    #     print("x.Shape" + str(x.shape))
    #     print("y.Shape" + str(y.shape))
    #     grad_w = np.dot(x.T, y_pred - y)/x.shape[0]
    #     print("grad_w" + str(grad_w))
    #     grad_b = np.sum(y_pred - y, axis=0)/x.shape[0]
    #     self.W -= self.lr * grad_w
    #     self.b -= self.lr * grad_b
    #     return np.dot(y_pred - y, self.W.T)
    

    def __str__(self):
        return "W: " + str(self.W) + " b: " + str(self.b)+ "activation: " + str(self.activation)
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
            tempExp = np.exp(x)
            tempSum = np.sum(tempExp, axis=0, keepdims=False)
            #divide each value in x by the sum of the row
            retVal = tempExp / tempSum
            return retVal
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, x):
        if self.activation == "relu":
            return x > 0
        elif self.activation == "sigmoid":
            return x * (1 - x)
        elif self.activation == "tanh":
            # return 1 - x**2
            return 1 - self.forward(x) ** 2
        elif self.activation == "softmax":
            # Todo check this
            # return x * (1 - x)
            selFor = self.forward(x)
            return selFor * (1 - selFor) 
            return self.forward(x) * (1 - self.forward(x))
    def __str__(self):
        return str(self.activation)
        
if __name__ == "__main__":
    X = np.random.rand(1, 2)
    y = np.random.rand(1, 2)

    y= np.array([[1,0]])
    nn = NN([2, 3, 2], lr=0.1)
    # print(nn.__str__()) 
    print(nn.predict(X.T))
    nn.train(X.T, y.T, 10)
    print(nn.predict(X.T))
    nn.train(X.T, y.T, 100)
    print(nn.predict(X.T))
    print(y)
    # print(nn.predict(X.T) - y.T)
    # print(np.sum(nn.predict(X.T) - y.T))