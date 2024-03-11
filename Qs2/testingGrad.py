from deep import * 
import numpy as np
import copy 
import os
import time
import matplotlib.pyplot as plt



def gradient_test_ploting_functions_byLecture(weights, biases, X, y, initial_epsilon=100, epsilon_iterator = [(0.5**(i)) for i in range(11)]):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits) #this is the base prediction
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred) #this is the base gradient
    grad_x= np.mean(np.dot(weights, (y_pred - y.T).T))
    base_loss = softmax_regression_loss(weights, biases, X, y, y_pred)
    print("y_pred", y_pred)

    grad_diffs_w = []
    grad_diffs_w_grad = []
    grad_diffs_b = []
    grad_diffs_b_grad = []
    grad_diffs_x = []
    grad_diffs_x_grad = []


    #starting from weights
    d = np.random.rand(weights.shape[0], weights.shape[1])
    #normalize the d vector
    d = d/np.linalg.norm(d)

    for epsilon in epsilon_iterator:
        
        weights_plus = weights.copy()
        weights_plus += epsilon*d
        print("weights_plus", weights_plus)
        y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
        print ("y_pred_plus", y_pred_plus)
        loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
        grad_w_plus = abs(loss_plus - base_loss)
        grad_diffs_w.append(copy.deepcopy( grad_w_plus))
        grad_diffs_w_grad.append(abs(loss_plus-base_loss- np.vdot(d,grad_w)*epsilon))



    #continue with biases
    d = np.random.rand(biases.shape[0], biases.shape[1])
    d = d/np.linalg.norm(d)
    for epsilon in epsilon_iterator:
        biases_plus = biases.copy()
        biases_plus += epsilon*d
        y_pred_plus = softmax(np.dot(X, weights) + biases_plus)
        loss_plus = softmax_regression_loss(weights, biases_plus, X, y, y_pred_plus)
        grad_b_plus = abs(loss_plus - base_loss)
        grad_diffs_b.append(copy.deepcopy( grad_b_plus))
        grad_diffs_b_grad.append(abs(loss_plus-base_loss- np.vdot(d,grad_b)*epsilon))


    #continue with x
    d = np.random.rand(X.shape[0], X.shape[1])
    d = d/np.linalg.norm(d)
    for epsilon in epsilon_iterator:
        X_plus = X.copy()
        X_plus += epsilon*d
        y_pred_plus = softmax(np.dot(X_plus, weights) + biases)
        loss_plus = softmax_regression_loss(weights, biases, X_plus, y, y_pred_plus)
        grad_x_plus = abs(loss_plus - base_loss)
        grad_diffs_x.append(copy.deepcopy( grad_x_plus))
        grad_diffs_x_grad.append(abs(loss_plus-base_loss- np.vdot(d,grad_x)*epsilon))

        

    plt.plot(grad_diffs_w,label= "loss difference")
    plt.plot(grad_diffs_w_grad, label="loss difference with grad")
    plt.yscale('log')
    plt.ylabel('Loss Difference in Log Scale')
    plt.xlabel('power of 0.5 for epsilon')
    plt.title('Loss Difference vs. Epsilon addition to W')
    plt.legend()
    plt.show()

    plt.plot(grad_diffs_b,label= "loss difference")
    plt.plot(grad_diffs_b_grad, label="loss difference with grad")
    plt.yscale('log')
    plt.ylabel('Loss Difference in Log Scale')
    plt.xlabel('power of 0.5 for epsilon')
    plt.title('Loss Difference vs. Epsilon addition to B')
    plt.legend()
    plt.show()

    
    plt.plot(grad_diffs_x,label= "loss difference")
    plt.plot(grad_diffs_x_grad, label="loss difference with grad")
    plt.yscale('log')
    plt.ylabel('Loss Difference in Log Scale')
    plt.xlabel('power of 0.5 for epsilon')
    plt.title('Loss Difference vs. Epsilon addition to X')
    plt.legend()
    plt.show()

    



    

# Example usage
np.random.seed(42)
X = np.random.rand(1, 1) #100 data points, each 1 features
y = np.eye(5, 1)  # Assuming 5 classes for softmax regression , one-hot encoding
weights = np.random.rand(1, 5) # 1 feature, 3 classes
biases = np.random.rand(1,5)
print(weights)
# Perform the gradient test

gradient_test_ploting_functions_byLecture(weights, biases, X, y, initial_epsilon=1, epsilon_iterator = [(0.5**(i)) for i in range(11)])