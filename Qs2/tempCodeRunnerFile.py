d = np.random.rand(biases.shape[0])
    # d = d/np.linalg.norm(d)
    # for epsilon in epsilon_iterator:
    #     biases_plus = biases.copy()
    #     biases_plus += epsilon*d
    #     y_pred_plus = softmax(np.dot(X, weights) + biases_plus)
    #     loss_plus = softmax_regression_loss(weights, biases_plus, X, y, y_pred_plus)
    #     grad_b_plus = abs(loss_plus - base_loss)
    #     grad_diffs_b.append(copy.deepcopy( grad_b_plus))
    #     grad_diffs_b_grad.append(abs(loss_plus-base_loss- np.vdot(d,grad_b)*epsilon))
    # #plot a graph with both grad_diffs_b and grad_diffs_b_grad on the same graph
    # #x axis is the epsilons power(from 0 to 10)
    # plt.plot(grad_diffs_b,label= "loss difference")
    # plt.plot(grad_diffs_b_grad, label="loss difference with grad")
    # plt.yscale('log')
    # plt.ylabel('Loss Difference in Log Scale')
    # plt.xlabel('power of 0.5 for epsilon')
    # plt.legend()
    # plt.show()
