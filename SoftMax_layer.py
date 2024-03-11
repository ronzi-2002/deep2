#TODO delete this file
import numpy as np
import torch #only for tests

class SoftmaxModule:

    def __init__(self, n, l):
        # n is the length of input vector, l is number of classes
        self.W = np.random.randn(n, l)*0.5   # change this potentially
        self.B = np.zeros((1,l))#np.random.rand(1,l) * 0.5
        self.X = None # cache X
        self.C = None # cache C
        self.A = None # cache A(activation, i.e. softmax output)

    def softmax_forward(self, X,bias=True,cache = True):
        """

        :param X: X tensor of shape n,m where m is number of training examples
        :param bias: whether to add bias or not, just for testing
        :return: softmax output of shape l*m where l is the number of classes
        """

        # shape of W is n,l. shape of X is n,m.
        if bias:
            temp_X = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0)  # shape n+1,m
            temp = temp_X.T @ np.concatenate((self.W,self.B),axis=0)  # shape of temp is m,l
        else:
            temp = X.T @ self.W #shape is m,l
        temp_max = np.max(temp, axis=1, keepdims=True) # shape is m,1
        temp = temp - temp_max # broadcasting
        exp = np.exp(temp)  # shape of exp is m,l
        # print("shape of exp: %s" % str(exp.shape))
        sum = np.sum(exp, axis=1, keepdims=True)  # shape of sum is m,1
        # print("shape of sum: %s" % str(sum.shape))
        out = exp / sum  # shape of out is m,l
        if cache:
          self.A = out # cache A
        return out

    def cross_entropy_forward(self, X, C):
        """

        :param X: output of softmax_forward, shape m,l
        :param C: matrix of true labels, shape l,m
        :return: cross entropy loss over all training examples
        """
        t = C.T * np.log(X)  # shape m,l
        l_sum = np.sum(t, axis=1)  # shape m
        return -1 * np.mean(l_sum)

    def loss_forward(self, X, C,cache = True):
        """

        :param X: input matrix of size n*m
        :param C: matrix of true labels, shape l,m
        :return: the function returns the cross entropy loss
        """
        ##CACHE X and C
        if cache:
          self.X = X
          self.C = C
        return self.cross_entropy_forward(self.softmax_forward(X,cache = cache), C)

    def grad(self):
        """

        :return: (dX,dW,dB)

        """
        # A.shape is m,l so we need to transpose it when computing dx
        # W.shape is n,l
        # X.shape is n,m
        m = self.X.shape[1]
        dX = (1/m)*self.W @ (self.A.T-self.C) # shape n,m
        temp_X = np.concatenate((self.X, np.ones((1, self.X.shape[1]))), axis=0)  # shape n+1,m. this is for bias
        dWB = (1/m)*temp_X @ (self.A-self.C.T) # shape n+1,l
        # now we need to separate dW and dB
        dW = dWB[:-1,:]
        dB = dWB[-1:,:]
        return dX,dW,dB # TODO: jacobian test

    # def grad_test(self):
    #     X = self.X
    #     C = self.C
    #     epsilon = [(0.5)**i for i in range(0,10)]
    #     o_eps = []
    #     o_eps_squared = []
    #     dX,dW,dB= self.grad()

    #     ### FOR dX ####
    #     d = np.random.rand(self.X.shape[0],self.X.shape[1])
    #     d = d/np.linalg.norm(d)
    #     for eps in epsilon:
    #         f_x_eps = self.loss_forward(X+(d*eps),C,cache=False)
    #         f_x = self.loss_forward(X,C,cache=False)
    #         o_eps.append(np.abs(f_x_eps-f_x))
    #         temp = np.vdot(d,dX)*eps
    #         temp_o_squared = np.abs(f_x_eps-f_x-temp)
    #         o_eps_squared.append(temp_o_squared)
    #     #now plot both o_eps and o_eps_squared
    #     # plt.plot()
    #     plt.plot(o_eps,label="without gradient",color="red")
    #     plt.plot(o_eps_squared,label="with gradient",color="blue")
    #     plt.yscale("log")
    #     plt.ylabel("difference")
    #     plt.legend()
    #     plt.title("softmax: dL/dX")
    #     plt.show()

    #     ##FOR W ##
    #     o_eps = []
    #     o_eps_squared = []
    #     d = np.random.rand(self.W.shape[0],self.W.shape[1])
    #     d = d/np.linalg.norm(d)
    #     for eps in epsilon:
    #         temp = self.W
    #         self.W = self.W+d*eps
    #         f_x_eps = self.loss_forward(X,C,cache=False)
    #         self.W = temp
    #         f_x = self.loss_forward(X,C,cache=False)
    #         # torch_test = ce(torch.Tensor(X.T @ self.W),torch.Tensor(C.T))
    #         temp_o = np.abs(f_x_eps-f_x)
    #         o_eps.append(np.abs(f_x_eps-f_x))
    #         temp = np.vdot(d,dW)*eps
    #         temp_o_squared = np.abs(f_x_eps-f_x-temp)
    #         # print("o(eps): %s, o(eps)^2: %s, epsilon: %s" % (str(temp_o),str(temp_o_squared),str(eps)))
    #         o_eps_squared.append(temp_o_squared)
    #     #now plot both o_eps and o_eps_squared
    #     # plt.plot()
    #     plt.plot(o_eps,label="without gradient",color="red")
    #     plt.plot(o_eps_squared,label="with gradient",color="blue")
    #     plt.yscale("log")
    #     plt.ylabel("difference")
    #     plt.legend()
    #     plt.title("softmax: dL/dW")
    #     plt.show()
    #     ## FOR B ##
    #     o_eps = []
    #     o_eps_squared = []
    #     d = np.random.rand(self.B.shape[0],self.B.shape[1])
    #     d = d/np.linalg.norm(d)
    #     for eps in epsilon:
    #         temp = self.B
    #         self.B = self.B+d*eps
    #         f_x_eps = self.loss_forward(X,C,cache=False)
    #         self.B = temp
    #         f_x = self.loss_forward(X,C,cache=False)
    #         # torch_test = ce(torch.Tensor(X.T @ self.W),torch.Tensor(C.T))
    #         temp_o = np.abs(f_x_eps-f_x)
    #         o_eps.append(np.abs(f_x_eps-f_x))
    #         temp = np.vdot(d,dB)*eps
    #         temp_o_squared = np.abs(f_x_eps-f_x-temp)
    #         # print("o(eps): %s, o(eps)^2: %s, epsilon: %s" % (str(temp_o),str(temp_o_squared),str(eps)))
    #         o_eps_squared.append(temp_o_squared)
    #     #now plot both o_eps and o_eps_squared
    #     # plt.plot()
    #     plt.plot(o_eps,label="without gradient",color="red")
    #     plt.plot(o_eps_squared,label="with gradient",color="blue")
    #     plt.yscale("log")
    #     plt.ylabel("difference")
    #     plt.legend()
    #     plt.title("softmax: dL/dB")
    #     plt.show()



# if __name__ == "__main__":
#     # test softmax
#     n = 2
#     l = 5
#     m = 3
#     X = np.random.rand(n, m)
#     # np.eye(n_classes)[np.random.choice(n_classes, n_samples)]
#     C =np.eye(l)[np.random.choice(l, m)].T
#     print("X: %s" % str(X.shape))
#     print("C: %s" % str(C.shape))
#     # print("X: %s" % X)
#     # print("C: %s" % C)
#     sm = SoftmaxModule(n, l)
#     out = sm.softmax_forward(X,False)
#     print("our_softmax_out: %s" % str(out))
#     # loss = sm.cross_entropy_forward(out, C)
#     # print("loss: %s" % loss)
#     test = X.T @ sm.W
#     # print("test: %s" % str(test.shape))
#     sm_test = torch.nn.functional.softmax(torch.Tensor(test), dim=1)
#     # ce_test = torch.nn.functional.cross_entropy(torch.Tensor(sm_test), torch.Tensor(C))
#     print("sm_test_torch: %s" % str(sm_test))

#     ce = sm.cross_entropy_forward(out,C)
#     print("our_ce: %s" % ce)


#     # test cross entropy
#     ce_torch = torch.nn.CrossEntropyLoss()
#     ce_test = ce_torch(torch.Tensor(test), torch.Tensor(C.T))
#     print("ce_test_torch: %s" % ce_test)

#     # now let's test if the bias works
#     b = np.random.rand(1,l)
#     sm.B = b
#     test = test+b
#     sm_bias = sm.softmax_forward(X,bias=True)
#     print("sm_bias: %s" % str(sm_bias))
#     sm_test = torch.nn.functional.softmax(torch.Tensor(test), dim=1)
#     print("sm_bias_torch: %s" % str(sm_test))