from deep import * 
import numpy as np
import copy 
import os
import time
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
def gradient_test_by_lecture_notes(weights, biases, X, y, initial_epsilon=1e-2):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits)
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred)
    grad_diffs_w = []
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in [initial_epsilon/(10**i) for i in range(10)]:
        grad_w_plus = np.zeros_like(weights)
        grad_b_plus = np.zeros_like(biases)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights_plus = weights.copy()
                weights_plus[i, j] += epsilon
                y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
                loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
                grad_w_plus[i, j] = (loss_plus - softmax_regression_loss(weights, biases, X, y, y_pred))/epsilon
        for i in range(biases.shape[0]):
            biases_plus = biases.copy()
            biases_plus[i] += epsilon
            loss_plus = softmax_regression_loss(weights, biases_plus, X, y, y_pred_plus)
            grad_b_plus[i] = (loss_plus - softmax_regression_loss(weights, biases, X, y, y_pred))/epsilon
        grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diffs_w.append(copy.deepcopy( grad_diff_w))
        grad_diff_b = np.linalg.norm(grad_b - grad_b_plus)

        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")

    #plot the gradient difference points with respect to the epsilon
    plt.scatter([initial_epsilon/(10**i) for i in range(10)], grad_diffs_w)
    plt.title('Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Gradient Difference')
    plt.show()

#Gradient Verification By Lecture Notes
def gradient_test_by_lecture_notes_linear_dec(weights, biases, X, y, initial_epsilon=1):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits)
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred)
    grad_diffs_w = []
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in [initial_epsilon-(0.1*i) for i in range(10)]:
        grad_w_plus = np.zeros_like(weights)
        grad_b_plus = np.zeros_like(biases)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights_plus = weights.copy()
                weights_plus[i, j] += epsilon
                y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
                loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
                grad_w_plus[i, j] = (loss_plus - softmax_regression_loss(weights, biases, X, y, y_pred))/epsilon
        for i in range(biases.shape[0]):
            biases_plus = biases.copy()
            biases_plus[i] += epsilon
            loss_plus = softmax_regression_loss(weights, biases_plus, X, y, y_pred_plus)
            grad_b_plus[i] = (loss_plus - softmax_regression_loss(weights, biases, X, y, y_pred))/epsilon
        grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diffs_w.append(copy.deepcopy( grad_diff_w))
        grad_diff_b = np.linalg.norm(grad_b - grad_b_plus)

        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")


#Gradient Verification By Lecture Notes
def gradient_test_by_lecture_notes_linear_dec_2(weights, biases, X, y, initial_epsilon=100):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits)
    y_pred = logits
    print("y_pred", y_pred, "logits", logits)
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred)
    base_loss = softmax_regression_loss(weights, biases, X, y, y_pred)
    print("base_loss", base_loss)
    
    grad_diffs_w = []
    grad_diffs_b = []
    grad_diffs_x = []
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in [initial_epsilon-(0.1*i) for i in range(11)]:
        # grad_w_plus = np.zeros_like(weights)
        # grad_b_plus = np.zeros_like(biases)
        # for i in range(weights.shape[0]):
            # for j in range(weights.shape[1]):
        weights_plus = weights.copy()
        print("weights_plus_before", weights_plus)
        weights_plus += epsilon
        print("weights_plus_after", weights_plus) 

        # y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
        y_pred_plus = np.dot(X, weights_plus) + biases
        print ("y_pred_plus", y_pred_plus,"logits", np.dot(X, weights_plus) + biases)
        accuracy = 0
        for i in range(y.shape[0]):
            if np.argmax(y[i]) == np.argmax(y_pred_plus[i]):
                accuracy += 1
        print("accuracy", accuracy/y.shape[0])
        loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
        print("loss_plus", loss_plus)

        grad_w_plus = (loss_plus - base_loss)#/epsilon
        print("grad_w_plus", grad_w_plus)
        # for i in range(biases.shape[0]):
        biases_plus = biases.copy()
        biases_plus += epsilon
        loss_plus = softmax_regression_loss(
            weights, biases_plus, X, y, y_pred_plus)
        grad_b_plus = (loss_plus - base_loss)#/epsilon
        
        
        grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diffs_w.append(copy.deepcopy( grad_diff_w))
        grad_diff_b = np.linalg.norm(grad_b - grad_b_plus)
        grad_diffs_b.append(copy.deepcopy( grad_diff_b))


        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")




  #plot the gradient difference points with respect to the epsilon
    plt.scatter([initial_epsilon-(0.1*i) for i in range(11)], grad_diffs_w)
    # Create the directory if it doesn't exist
    directory = "grad_tests_Qs2/" + time.strftime("%Y%m%d-%H%M%S") + "_lin"
    os.makedirs(directory, exist_ok=True)

    # Save the plots instead of showing them
    plt.title('W Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('W Gradient Difference')
    plt.savefig(os.path.join(directory, 'w_gradient_difference.png'))
    #clear the plot
    plt.clf()

    plt.scatter([initial_epsilon-(0.1*i) for i in range(11)], grad_diffs_b)
    plt.title('b Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('b Gradient Difference')
    plt.savefig(os.path.join(directory, 'b_gradient_difference.png'))
    plt.clf()

    



#Gradient Verification By Lecture Notes
def gradient_test_by_lecture_notes_linear_exponantial(weights, biases, X, y, initial_epsilon=100):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits)
    y_pred = logits
    print("y_pred", y_pred, "logits", logits)
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred)
    base_loss = softmax_regression_loss(weights, biases, X, y, y_pred)
    print("base_loss", base_loss)
    
    grad_diffs_w = []
    grad_diff_w_d_grad = []
    grad_diffs_b = []
    grad_diffs_x = []
    d= np.random.rand(X.shape[0], X.shape[1])
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in [initial_epsilon/(10**i) for i in range(11)]:
        # grad_w_plus = np.zeros_like(weights)
        # grad_b_plus = np.zeros_like(biases)
        # for i in range(weights.shape[0]):
            # for j in range(weights.shape[1]):
        weights_plus = weights.copy()
        print("weights_plus_before", weights_plus)
        weights_plus += epsilon
        print("weights_plus_after", weights_plus) 

        # y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
        y_pred_plus = np.dot(X, weights_plus) + biases
        print ("y_pred_plus", y_pred_plus,"logits", np.dot(X, weights_plus) + biases)
        accuracy = 0
        for i in range(y.shape[0]):
            if np.argmax(y[i]) == np.argmax(y_pred_plus[i]):
                accuracy += 1
        print("accuracy", accuracy/y.shape[0])
        loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
        print("loss_plus", loss_plus)

        grad_w_plus = (loss_plus - base_loss)/epsilon
        
        print("grad_w_plus", grad_w_plus)
        # for i in range(biases.shape[0]):
        biases_plus = biases.copy()
        biases_plus += epsilon
        loss_plus = softmax_regression_loss(
            weights, biases_plus, X, y, y_pred_plus)
        grad_b_plus = (loss_plus - base_loss)/epsilon
        
        
        grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diffs_w.append(copy.deepcopy( grad_diff_w))
        grad_diff_b = np.linalg.norm(grad_b - grad_b_plus)
        grad_diffs_b.append(copy.deepcopy( grad_diff_b))


        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")




    #plot the gradient difference points with respect to the epsilon
    plt.scatter([initial_epsilon/(10**i) for i in range(11)], grad_diffs_w)
    # Create the directory if it doesn't exist
    directory = "grad_tests_Qs2/" + time.strftime("%Y%m%d-%H%M%S")+"_exp"
    os.makedirs(directory, exist_ok=True)

    # Save the plots instead of showing them
    plt.title('W Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('W Gradient Difference')
    plt.savefig(os.path.join(directory, 'w_gradient_difference.png'))
    #clear the plot
    plt.clf()

    plt.scatter([initial_epsilon/(10**i) for i in range(11)], grad_diffs_b)
    plt.title('b Gradient Difference vs. Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('b Gradient Difference')
    plt.savefig(os.path.join(directory, 'b_gradient_difference.png'))
    plt.clf()

def JacMV(x,v,weights,biases,X,y,y_pred):
    #computes the multiplication of the Jacobian with a vector v. The derivative is computed at the point x
    # x: the point at which the derivative is computed
    # v: the vector to multiply the Jacobian with

    #compute the Jacobian
    J = np.zeros((y.shape[1],weights.shape[0],weights.shape[1]))
    for i in range(y.shape[1]):
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                weights_plus = weights.copy()
                weights_plus[j, k] += x
                y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
                print("x", x[0])
                J[i,j,k] = (y_pred_plus[i]-y_pred[i])/x[0]
    #multiply the Jacobian with the vector v
    result = np.zeros_like(v)
    for i in range(y.shape[1]):
        result += np.dot(J[i],v[i])
    return result



def gradient_test_ploting_functions(weights, biases, X, y, initial_epsilon=100, epsilon_iterator = [1/(10**i) for i in range(11)]):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits)
    y_pred = logits
    print("y_pred", y_pred, "logits", logits)
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred)
    base_loss = softmax_regression_loss(weights, biases, X, y, y_pred)
    print("base_loss", base_loss)
    
    grad_diffs_w = []
    grad_diff_w_d_grad = []
    grad_diffs_b = []
    grad_diffs_b_d_grad = []
    grad_diffs_x = []
    grad_diff_w_d_Jacobian = []
    grad_diffs_b_d_Jacobian = []
    #d is a random vector the same size as weights
    d_w= np.random.rand(weights.shape[0], weights.shape[1])
    d_b= np.random.rand(biases.shape[0])
    #compute the gradient with adding a decreasing epsilon to the weights
    for epsilon in epsilon_iterator:
        # grad_w_plus = np.zeros_like(weights)
        # grad_b_plus = np.zeros_like(biases)
        # for i in range(weights.shape[0]):
            # for j in range(weights.shape[1]):
        weights_plus = weights.copy()
        print("weights_plus_before", weights_plus)
        weights_plus += epsilon
        print("weights_plus_after", weights_plus) 

        # y_pred_plus = softmax(np.dot(X, weights_plus) + biases)
        y_pred_plus = np.dot(X, weights_plus) + biases
        print ("y_pred_plus", y_pred_plus,"logits", np.dot(X, weights_plus) + biases)
        accuracy = 0
        for i in range(y.shape[0]):
            if np.argmax(y[i]) == np.argmax(y_pred_plus[i]):
                accuracy += 1
        print("accuracy", accuracy/y.shape[0])
        loss_plus = softmax_regression_loss(weights_plus, biases, X, y, y_pred_plus)
        print("loss_plus", loss_plus)

        grad_w_plus = abs(loss_plus - base_loss)#/epsilon
        grad_diff_w_d_grad_plus = abs(loss_plus-base_loss- np.dot(grad_w,epsilon*d_w.T))
        grad_diff_w_d_Jacobian_plus = abs(loss_plus-base_loss- JacMV(X[0],epsilon*d_w,weights,biases,X,y,y_pred))

        print("aaaaaaaaaaaaaaaaaaaaaagrad_diff_w_d_grad_plus", grad_diff_w_d_grad_plus)
        print("grad_w_plus", grad_w_plus)
        # for i in range(biases.shape[0]):
        biases_plus = biases.copy()
        biases_plus += epsilon
        loss_plus = softmax_regression_loss(
            weights, biases_plus, X, y, y_pred_plus)
        grad_b_plus = abs(loss_plus - base_loss)#/epsilon
        grad_diff_b_d_grad_plus = abs(loss_plus-base_loss- np.dot(grad_b,epsilon*d_b.T))
        grad_diff_b_d_Jacobian_plus = abs(loss_plus-base_loss- JacMV(X[0],epsilon*d_b,weights,biases,X,y,y_pred))
        
        # grad_diff_w = np.linalg.norm(grad_w - grad_w_plus)
        grad_diff_w = grad_w_plus
        grad_diffs_w.append(copy.deepcopy( grad_diff_w))
        grad_diff_b = grad_b_plus
        grad_diffs_b.append(copy.deepcopy( grad_diff_b))
        grad_diff_w_d_grad.append(copy.deepcopy( grad_diff_w_d_grad_plus))
        grad_diffs_b_d_grad.append(copy.deepcopy( grad_diff_b_d_grad_plus))
        grad_diff_w_d_Jacobian.append(copy.deepcopy( grad_diff_w_d_Jacobian_plus))
        grad_diffs_b_d_Jacobian.append(copy.deepcopy( grad_diff_b_d_Jacobian_plus))



        print("Gradient test results:")
        print(f"Weight gradient difference: {grad_diff_w}")
        print(f"Bias gradient difference: {grad_diff_b}")



    print("size of grad_diffs_w_d_grad", len(grad_diff_w_d_grad))
    print("grad_diffs_w_d_grad", grad_diff_w_d_grad[0])
    #plot the gradient difference points with respect to the epsilon
    plt.scatter( epsilon_iterator, grad_diffs_w)
    # plt.plot([initial_epsilon/(10**i) for i in range(11)], np.ravel(grad_diff_w_d_grad))
    # Create the directory if it doesn't exist
    directory = "grad_tests_Qs2/" + time.strftime("%Y%m%d-%H%M%S")+"_exp"
    os.makedirs(directory, exist_ok=True)

    # Save the plots instead of showing them
    plt.title('Loss Difference vs. Epsilon addition to W and grad')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.legend(["grad_diff_w", "grad_diff_w_d_grad"])
    plt.savefig(os.path.join(directory, 'w_gradient_difference.png'))
    #clear the plot
    plt.clf()

    plt.scatter(epsilon_iterator, grad_diffs_b)
    plt.title('Loss Difference vs. Epsilon addition to B and grad')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.savefig(os.path.join(directory, 'b_gradient_difference.png'))
    plt.clf()
    
    plt.scatter(epsilon_iterator,grad_diff_w_d_grad)
    plt.title('Loss Difference vs. Epsilon addition to W ')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.savefig(os.path.join(directory, 'w_gradient_difference_d_grad.png'))
    plt.clf()

    plt.scatter(epsilon_iterator,  np.ravel(grad_diffs_b_d_grad))
    plt.title('Loss Difference vs. Epsilon addition to B ')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.savefig(os.path.join(directory, 'b_gradient_difference_d_grad.png'))
    plt.clf()

    plt.scatter(epsilon_iterator, grad_diff_w_d_Jacobian)
    plt.title('Loss Difference vs. Epsilon addition to W and Jacobian')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.savefig(os.path.join(directory, 'w_gradient_difference_Jacobian.png'))
    plt.clf()

    plt.scatter(epsilon_iterator, grad_diffs_b_d_Jacobian)
    plt.title('Loss Difference vs. Epsilon addition to B and Jacobian')
    plt.xlabel('Epsilon')
    plt.ylabel('Loss Difference')
    plt.savefig(os.path.join(directory, 'b_gradient_difference_Jacobian.png'))
    plt.clf()




def gradient_test_ploting_functions_byLecture(weights, biases, X, y, initial_epsilon=100, epsilon_iterator = [(0.5**(i)) for i in range(11)]):
    # Compute gradients using the provided function
    logits = np.dot(X, weights) + biases
    y_pred = softmax(logits) #this is the base prediction
    grad_w, grad_b = compute_grad(weights, biases, X, y,y_pred) #this is the base gradient
    grad_x= np.mean(np.dot(weights, (y_pred - y).T))
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
    #plot a graph with both grad_diffs_w and grad_diffs_w_grad on the same graph
    #x axis is the epsilons power(from 0 to 10)
    plt.plot(grad_diffs_w,label= "loss difference")
    plt.plot(grad_diffs_w_grad, label="loss difference with grad")
    plt.yscale('log')
    plt.legend()
    plt.show()

    



    

# Example usage
# Define your data (X, y), weights, and biases
np.random.seed(42)
# (n,m)

X = np.random.rand(1, 1) #100 data points, each 1 features
# (l,m)
y = np.eye(1, 5)  # Assuming 5 classes for softmax regression , one-hot encoding
weights = np.random.rand(1, 5) # 1 feature, 3 classes
biases = np.random.rand(1,5)
print(weights)
# Perform the gradient test

# gradient_test(weights, biases, X, y)

# gradient_test_by_lecture_notes_linear_dec_2(weights, biases, X, y, initial_epsilon=1)
# gradient_test_by_lecture_notes_linear_exponantial(weights, biases, X, y, initial_epsilon=1)
# gradient_test_ploting_functions(weights, biases, X, y, initial_epsilon=1, epsilon_iterator = [1/(10**i) for i in range(11)])
# gradient_test_ploting_functions(weights, biases, X, y, initial_epsilon=1, epsilon_iterator = [1/(10**i) for i in range(11)])
gradient_test_ploting_functions_byLecture(weights, biases, X, y, initial_epsilon=1, epsilon_iterator = [(0.5**(i)) for i in range(11)])