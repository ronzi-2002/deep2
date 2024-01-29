from deep import * 
import numpy as np
import matplotlib.pyplot as plt



# Gradient test function
def gradient_test(weights, biases, X, y, epsilon=1e-8):
    # Compute gradients using the provided function
    grad_w, grad_b = compute_grad(weights, biases, X, y)

    # Numerical gradient approximation
    numerical_grad_w = np.zeros_like(weights)
    numerical_grad_b = np.zeros_like(biases)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights_plus = weights.copy()
            weights_minus = weights.copy()

            weights_plus[i, j] += epsilon
            weights_minus[i, j] -= epsilon

            loss_plus = softmax_regression_loss(weights_plus, biases, X, y)
            loss_minus = softmax_regression_loss(weights_minus, biases, X, y)

            numerical_grad_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    for i in range(biases.shape[0]):
        biases_plus = biases.copy()
        biases_minus = biases.copy()

        biases_plus[i] += epsilon
        biases_minus[i] -= epsilon

        loss_plus = softmax_regression_loss(weights, biases_plus, X, y)
        loss_minus = softmax_regression_loss(weights, biases_minus, X, y)

        numerical_grad_b[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare gradients
    grad_diff_w = np.linalg.norm(grad_w - numerical_grad_w)
    grad_diff_b = np.linalg.norm(grad_b - numerical_grad_b)

    print("Gradient test results:")
    print(f"Weight gradient difference: {grad_diff_w}")
    print(f"Bias gradient difference: {grad_diff_b}")

#Gradient Verification By Lecture Notes
def gradient_test_by_lecture_notes(weights, biases, X, y, initial_epsilon=1e-4):
    # Compute gradients using the provided function
    grad_w, grad_b = compute_grad(weights, biases, X, y)
    grad_diffs_w = []
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in [initial_epsilon/(10**i) for i in range(5)]:
        grad_w_plus = np.zeros_like(weights)
        grad_b_plus = np.zeros_like(biases)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights_plus = weights.copy()
                weights_plus[i, j] += epsilon
                loss_plus = softmax_regression_loss(weights_plus, biases, X, y)
                grad_w_plus[i, j] = (loss_plus - softmax_regression_loss(weights, biases, X, y))/epsilon
        for i in range(biases.shape[0]):
            biases_plus = biases.copy()
            biases_plus[i] += epsilon
            loss_plus = softmax_regression_loss(weights, biases_plus, X, y)
            grad_b_plus[i] = (loss_plus - softmax_regression_loss(weights, biases, X, y))/epsilon
        grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diffs_w.append(grad_diff_w)
        grad_diff_b = np.linalg.norm(grad_b - grad_b_plus)

        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")

    #plot the gradient difference points with respect to the epsilon
    plt.scatter([initial_epsilon/(10**i) for i in range(5)], grad_diffs_w)
    plt.title('Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Gradient Difference')
    plt.show()


# Example usage
# Define your data (X, y), weights, and biases
X = np.random.rand(100, 5) #100 data points, each 5 features
y = np.eye(100, 3)  # Assuming 3 classes for softmax regression , one-hot encoding
weights = np.random.rand(5, 3) #5 features, 3 classes
biases = np.random.rand(3) #3 classes
print(weights)
# Perform the gradient test


# gradient_test(weights, biases, X, y)

gradient_test_by_lecture_notes(weights, biases, X, y, initial_epsilon=1e-8)