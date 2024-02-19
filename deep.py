'''
Write the code for computing the loss function “soft-max regression” and its gradient
with respect to wj and the biases. Make sure that the derivatives are correct using the
gradient test (See the subsection “Gradient and Jacobian Verification” in the notes).
You should demonstrate and submit the results of the gradient test.
'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
'''implement the cross entropy loss function'''
def cross_entropy_loss_single(y, y_hat):#for a single data point
    epsilon = 1e-10  # Small constant to avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.sum(y*np.log(y_hat)) # y_hat is the predicted value, y is the true value

def cross_entropy_loss_batch(y, y_hat):#for a batch of data points
    epsilon = 1e-10  # Small constant to avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.sum(y*np.log(y_hat))/y.shape[0] # y_hat is the predicted value, y is the true value

'''implement the softmax function'''
def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)

#calculating the loss function for softmax regression given the weights and biases
def softmax_regression_loss(weights, biases, X, y, y_pred):
    # logits = np.dot(X, weights) + biases
    # y_pred = softmax(logits)
    loss = cross_entropy_loss_batch(y, y_pred)
    return loss

#calculating the gradient of the loss function for softmax regression given the weights and biases
def compute_grad(weights, biases, X, y, y_pred):
    # logits = np.dot(X, weights) + biases
    # y_pred = softmax(logits)
    # print ("y_pred" + str(y_pred))
    grad_w = np.dot(X.T, y_pred - y)/X.shape[0]
    #if minimal value of y_pred -y is positive, print it
    # if np.min(y_pred - y) > 0:
    # print("y_pred - y" + str(np.min(y_pred - y))+str(np.max(y_pred - y)))
    grad_b = np.sum(y_pred - y, axis=0)/X.shape[0]
    # print(grad_w)
    return grad_w, grad_b

def compute_grad_for_MSE(weights, biases, X, y, y_pred):
    m = len(X)
    
    # Compute gradients for weights
    grad_weights = (-2/m) * np.dot(X.T, (y - y_pred))
    
    # Compute gradient for biases
    grad_biases = (-2/m) * np.sum(y - y_pred)
    
    return grad_weights, grad_biases

#Gradient and Jacobian Verification
# 2.2 implement SGD
def sgd(weights, biases, X, y,loss_function=cross_entropy_loss_batch,gradient_function=compute_grad, learning_rate=0.1, num_iters=100, batch_size=100):
    losses = []
    all_weights = []

    for i in range(num_iters):
        # Randomly sample a batch of data points
        indices = np.random.permutation(X.shape[0])
        X_train = X[indices]
        y_train = y[indices]
        batchesxTrain = [X_train[i:i + batch_size] for i in range(0, X_train.shape[0], batch_size)]
        batchesyTrain = [y_train[i:i + batch_size] for i in range(0, y_train.shape[0], batch_size)]
        for j in range(len(batchesxTrain)):
            X_batch = batchesxTrain[j]
            y_batch = batchesyTrain[j]
            # Compute gradients
            logits = np.dot(X_batch, weights) + biases
            # print("logits" + str(logits))
            # if there is only one output, we don't need to use softmax
            y_pred = logits
            if len(y_batch[0]) > 1:
                y_pred = softmax(logits)   

            grad_w, grad_b = gradient_function(weights, biases, X_batch, y_batch, y_pred)
            # Update weights
            all_weights.append(deepcopy(weights))
            # print("weight before update" + str(weights), "grad_w" + str(grad_w))
            weights -= learning_rate * grad_w
            # print("weight after update" + str(weights))
            biases -= learning_rate * grad_b
            # Compute loss
            loss = loss_function(y_batch, y_pred)
            losses.append(loss)


        # # Plot the data and solutions
        # plt.scatter(X, y, label="True Data")
        # # plt.plot(X, X_b.dot(theta_closed_form), label="Closed-form Solution", color='green', linewidth=2)
        # plt.plot(X, X.dot(weights) + biases, label="SGD Solution", color='red', linestyle='dashed', linewidth=2)

        # plt.xlabel("X")
        # plt.ylabel("y")
        # plt.legend()
        # plt.title("Linear Regression - Closed-form vs SGD")
        # plt.show()





    # print(all_weights)
    return weights, biases, losses

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


#i need a function that receives an array of points and returns the slope and the bias of the line that fits the points
def closed_form_solution(X, y):
    # Add bias term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Compute closed-form solution
    theta_closed_form = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_closed_form



# demonstrate that the SGD works on a small least squares example
def test_sgd():
    # Generate random data points
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    #export the data to a xls file, having the first column as the x values and the second column as the y values
    # np.savetxt("data.csv", np.column_stack((X, y)), delimiter=",", fmt='%s')

        # Add bias term to X
    # X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # print(X)

    # Compute closed-form solution
    theta_closed_form = closed_form_solution(X, y)
    print("theta_closed_form")
    print(theta_closed_form)

    # Initialize weights and biases for SGD
    initial_weights = np.random.randn(1, 1)
    # print("initial_weights")
    # print(initial_weights)
    # print("initial_weights")
    initial_biases = np.random.randn(1)

    # Compute SGD solution
    final_weights, final_biases, losses = sgd(initial_weights, initial_biases, X, y,loss_function=mean_squared_error,gradient_function=compute_grad_for_MSE)#todo maybe use gradient for the mean squared error
    # final_weights, final_biases, losses = sgd(initial_weights, initial_biases, X, y,loss_function=cross_entropy_loss_batch)
    print("final_weights")
    print(final_weights)
    print("final_biases")
    print(final_biases)
    #plot the losses
    plt.plot(losses)
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


    # Plot the data and solutions
    plt.scatter(X, y, label="True Data")
    # plt.plot(X, X_b.dot(theta_closed_form), label="Closed-form Solution", color='green', linewidth=2)
    plt.plot(X, X.dot(final_weights) + final_biases, label="SGD Solution", color='red', linestyle='dashed', linewidth=2)
    plt.plot(X, X.dot(theta_closed_form[1]) + theta_closed_form[0], label="Closed-form Solution", color='green', linewidth=2)

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression - Closed-form vs SGD")
    plt.show()

    # Check if the solutions are close
    # assert np.allclose(theta_closed_form, np.concatenate([final_biases, final_weights[1:]]), rtol=1e-2), "SGD did not converge to the correct solution"

    # Predict using the closed-form solution
    y_pred_closed_form =  X.dot(theta_closed_form[1]) + theta_closed_form[0]
    # Predict using the SGD solution
    y_pred_sgd = X.dot(final_weights) + final_biases

    # Calculate and print mean squared errors
    mse_closed_form = mean_squared_error(y, y_pred_closed_form)
    mse_sgd = mean_squared_error(y, y_pred_sgd)
    print("Mean Squared Error (Closed-form):", mse_closed_form)
    print("Mean Squared Error (SGD):", mse_sgd)

if __name__ == "__main__":
    # Run the test and plot
    test_sgd()