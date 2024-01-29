'''
Write the code for computing the loss function “soft-max regression” and its gradient
with respect to wj and the biases. Make sure that the derivatives are correct using the
gradient test (See the subsection “Gradient and Jacobian Verification” in the notes).
You should demonstrate and submit the results of the gradient test.
'''
import numpy as np
import matplotlib.pyplot as plt
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
    grad_w = np.dot(X.T, y_pred - y)/X.shape[0]
    grad_b = np.sum(y_pred - y, axis=0)/X.shape[0]
    return grad_w, grad_b

#Gradient and Jacobian Verification
# 2.2 implement SGD
def sgd(weights, biases, X, y,loss_function=cross_entropy_loss_batch, learning_rate=0.00001, num_iters=10, batch_size=10):
    losses = []
    all_weights = []

    for i in range(num_iters):
        # Randomly sample a batch of data points
        batch_indices = np.random.randint(0, X.shape[0], size=batch_size)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        # Compute gradients
        logits = np.dot(X_batch, weights) + biases#TODO: we might need to change this to X_batch
        y_pred = softmax(logits)

        grad_w, grad_b = compute_grad(weights, biases, X_batch, y_batch, y_pred)
        # Update weights
        all_weights.append(weights)
        weights -= learning_rate * grad_w
        biases -= learning_rate * grad_b
        # Compute loss
        loss = loss_function(X_batch, y_pred)#TODO: we might need to change this to X_batch
        losses.append(loss)
    print(all_weights)
    return weights, biases, losses

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
def closed_form_solution(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    #print the slope 
    print(theta_best)
    
    return theta_best
# demonstrate that the SGD works on a small least squares example
def test_sgd():
    # Generate random data points
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

        # Add bias term to X
    # X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # print(X)

    # Compute closed-form solution
    theta_closed_form = closed_form_solution(X, y)

    # Initialize weights and biases for SGD
    initial_weights = np.random.randn(1, 1)
    print(initial_weights)
    initial_biases = np.random.randn(1)

    # Compute SGD solution
    final_weights, final_biases, losses = sgd(initial_weights, initial_biases, X, y,loss_function=mean_squared_error)
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

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression - Closed-form vs SGD")
    plt.show()

    # Check if the solutions are close
    assert np.allclose(theta_closed_form, np.concatenate([final_biases, final_weights[1:]]), rtol=1e-2), "SGD did not converge to the correct solution"

    # Predict using the closed-form solution
    y_pred_closed_form = X_b.dot(theta_closed_form)

    # Predict using the SGD solution
    y_pred_sgd = X_b.dot(np.concatenate([final_biases, final_weights[1:]]))

    # Calculate and print mean squared errors
    mse_closed_form = mean_squared_error(y, y_pred_closed_form)
    mse_sgd = mean_squared_error(y, y_pred_sgd)
    print("Mean Squared Error (Closed-form):", mse_closed_form)
    print("Mean Squared Error (SGD):", mse_sgd)

# Run the test and plot
test_sgd()