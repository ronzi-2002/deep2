#implementing a simple neural network
from Qs2.deep import cross_entropy_loss_batch
import numpy as np
import matplotlib.pyplot as plt
class NN:
    def __init__(self, dims, lr=0.1,isResNet=False):
        
        
        if not isResNet:  
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
                    self.layers.append(Layer(dims[i-1], dims[i], Activation("tanh"), lr=lr, layerNum=i-1))
                else:
                    self.layers.append(Layer(dims[i-1], dims[i], Activation("softmax"), lr=lr, layerNum=i-1))
            self.loss= None
        else:
            #the dims first value is the input dimension,  all the other values are the dimensions of the inner layers and the last value is the output dimension
            self.layers = []
            dim = dims[0]
            for i in range(1, len(dims)):
                if i != len(dims) - 1:
                    self.layers.append(ResiduaBlock(dim, dims[i], Activation("tanh"), lr=lr))
                else:
                    self.layers.append(Layer(dim, dims[i], Activation("softmax"), lr=lr))
    def forward(self, x,y  = None):
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
            y_pred, loss = self.forward(X,y)
            self.loss = loss
            self.backward()
            print("Epoch: " + str(epoch) + " Loss: " + str(loss))
    def predict(self, X):
        return self.forward(X)
    
    def __str__(self):
        for layer in self.layers:
            print(layer)
    
    def Jacobian_Test(self):
        for layer in self.layers[:-1]:
            print("Jacobian: " + str(layer.Jacobian_Test()))


class Layer:
    def __init__(self, input_dim, output_dim,activation, lr=0.1, layerNum=0):
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
        
    def update_weights(self, dx_from_next_layer):#TODO modify this(in resnet too)
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
    
    def set_param(self, name, value):
        if name == 'W':
            self.W = value
        elif name == 'b':
            self.b = value
        elif name == 'X':
            self.lastInput = value

        else:
            print("Parameter not found")
    def __str__(self):
        return "W: " + str(self.W) + " b: " + str(self.b)+ "activation: " + str(self.activation)
    def Jacobian_Test(self):
        #TODO refine this
        epsilon = [(0.5) ** i for i in range(0, 10)]
        u = np.random.rand(self.W.shape[0], self.lastInput.shape[1])  
        dW, dB, dX= self.gradient(u)
        parameters = [(self.lastInput, "dX", dX, 'X'), (self.W, "dW", dW, 'W'), (self.b, "dB", dB, 'b')]
        self.perform_jac_test(parameters, epsilon, u, "1")


   


    def perform_jac_test(self, parameters, epsilon, u, layer_title):
        for (param, param_name, gradient, name) in parameters:
            o_eps = []
            o_eps_squared = []

            d = np.random.rand(*param.shape)
            d = d / np.linalg.norm(d)

            for eps in epsilon:
                temp = param.copy()
                self.set_param(name, param + d * eps)
                afterForward = self.forward(self.lastInput)
                f_x_eps = np.vdot(u, self.forward(self.lastInput))
                self.set_param(name, temp)
                f_x = np.vdot(u, self.forward(self.lastInput))

                o_eps.append(np.abs(f_x_eps - f_x))

                temp_gradient = np.vdot(d, gradient) * eps
                temp_o_squared = np.abs(f_x_eps - f_x - temp_gradient)
                o_eps_squared.append(temp_o_squared)

            plt.plot(o_eps, label="without gradient", color="red")
            plt.plot(o_eps_squared, label="with gradient", color="blue")
            plt.yscale("log")
            # plt.xscale("log")
            # plt.xlabel("epsilon")
            plt.ylabel("difference")
            plt.legend()
            plt.title("title")
            plt.show()

        
class ResiduaBlock:
    def __init__(self, input_dim, inner_dim,activation, lr=0.1):
        self.layers = []
        self.layers.append(Layer(input_dim, inner_dim, activation, lr=lr))
        self.layers.append(Layer(inner_dim, input_dim, activation, lr=lr))
        self.lr = lr
        self.activation = activation
        self.lastForwardValAfterActivation = None
        self.lastForwardValBeforeActivation = None
        self.lastInput = None
    
    def forward(self, x):
        # pretty straight forward
        self.lastInput = x
        self.lastForwardValBeforeActivation = self.layers[1].forward(self.layers[0].forward(x))+x
        self.lastForwardValAfterActivation = self.activation.forward(self.lastForwardValBeforeActivation)
        return self.lastForwardValAfterActivation
    
    def gradient(self, dx_from_next_layer):

        activation_grad = self.activation.backward(self.lastForwardValBeforeActivation)
        grad_w2 = dx_from_next_layer * np.dot(activation_grad, self.layers[0].lastForwardValAfterActivation.T)
        grad_b2 = np.sum(dx_from_next_layer * activation_grad, axis=1, keepdims=True)
        dA = np.dot(self.layers[1].W.T,(dx_from_next_layer * activation_grad))
        activation_grad = self.activation.backward(self.layers[0].lastForwardValBeforeActivation)
        grad_w1 = dA * np.dot(activation_grad, self.lastInput.T)
        grad_b1 = np.sum(dA * activation_grad, axis=1, keepdims=True)
        current_dx = np.dot( self.layers[0].W.T,(dA * activation_grad))+dx_from_next_layer*self.activation.backward(self.lastForwardValBeforeActivation)
        return grad_w1, grad_b1, grad_w2, grad_b2, current_dx
    
    def update_weights(self, dx_from_next_layer):
        grad_w1, grad_b1, grad_w2, grad_b2, current_dx = self.gradient(dx_from_next_layer)
        self.layers[0].W -= self.lr * grad_w1
        self.layers[0].b -= self.lr * grad_b1
        self.layers[1].W -= self.lr * grad_w2
        self.layers[1].b -= self.lr * grad_b2
        return current_dx
    
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
            # print("x: " + str(x))
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



def runJacTest():
    X = np.random.rand(1, 2)
    y = np.random.rand(1, 2)

    y= np.array([[1,0]])
    nn = NN([2, 3, 2], lr=0.1)
    nn.forward(X.T,y)
    nn.Jacobian_Test()

if __name__ == "__main__":
    X = np.random.rand(1, 2)
    y = np.random.rand(1, 2)

    y= np.array([[1,0]])
    nn = NN([2, 3, 2], lr=0.1)
    
    # print(nn.__str__()) 
    # print(nn.predict(X.T))
    nn.train(X.T, y.T, 10)
    # print(nn.predict(X.T))
    # nn.train(X.T, y.T, 100)
    # print(nn.predict(X.T))
    # print(y)

    # print(nn.predict(X.T) - y.T)
    # print(np.sum(nn.predict(X.T) - y.T))

    # resNet = NN([2, 3, 2], lr=0.1, isResNet=True)
    # resNet.train(X.T, y.T, 10)
    # print(resNet.predict(X.T))
    # resNet.train(X.T, y.T, 100)
    # print(resNet.predict(X.T))
    # print(y)

    nn = NN([2, 3, 2], lr=0.1)
    nn.forward(X.T,y)
    nn.Jacobian_Test()

