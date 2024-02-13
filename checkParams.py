from deep import sgd, softmax
import numpy as np
import matplotlib.pyplot as plt

def computeAccuracy(X, y, weights, biases):
    # compute the accuracy of the model
    logits = np.dot(X, weights) + biases
    y_pred = logits
    if len(y[0]) > 1:
        y_pred = softmax(logits)
    correct = 0
    for i in range(y.shape[0]):
        if np.argmax(y[i]) == np.argmax(y_pred[i]):
            correct += 1
    return correct / y.shape[0]

def findBestParams(X_train, y_train, X_val, y_val):
    # find the best learning rate and best batch size
    bestLearningRate = 0
    bestBatchSize = 0
    bestAccuracy = 0
    epoch = 10
    for learningRate in [0.001, 0.01, 0.1, 1]:
        for batchSize in [10, 100, 1000]:
            weights = np.random.rand(X_train.shape[1], y_train.shape[1])
            biases = np.random.rand(y_train.shape[1])
            accuracyPerEpoch = []
            for i in range(epoch):
                weights, biases, loss = sgd(weights, biases, X_train, y_train, learning_rate=learningRate, num_iters=10, batch_size=batchSize)
                accuracy = computeAccuracy(X_val, y_val, weights, biases)
                accuracyPerEpoch.append(accuracy)
            plt.plot(range(epoch), accuracyPerEpoch)
            accuracy = computeAccuracy(X_val, y_val, weights, biases)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestLearningRate = learningRate
                bestBatchSize = batchSize
    # plot 
    return bestLearningRate, bestBatchSize