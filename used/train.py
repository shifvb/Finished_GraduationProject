"""
CS294A/CS294W Programming Assignment Starter Code

Instructions
------------

This file contains code that helps you get started on the
programming assignment. You will need to complete the code in sampleIMAGES.m,
sparseAutoencoderCost.m and computeNumericalGradient.m.
For the purpose of completing the assignment, you do not need to
change the code in this file.
"""
import used.gradient as gradient
import numpy as np
import scipy.optimize
from used.sparse_autoencoder import initialize, sparse_autoencoder_cost


def train(input_data: np.ndarray, visible_size: int, hidden_size: int, sparsity_param=0.1, lambda_=3e-3, beta=3,
          debug=False):
    """
    :param input_data: a numpy array of input data, should be of shape [visible_size, number_of_images]
    :param visible_size: number of input units
    :param hidden_size:  number of hidden units
    :param sparsity_param: desired average activation of the hidden units.
        (This was denoted by the Greek alphabet rho, which looks like a lower-case "p", in the lecture notes).
    :param lambda_: weight decay parameter
    :param beta: weight of sparsity penalty term
    :param debug: default to False, If True, will enable STEP 3(seed code of STEP 3)
    :return: theta: All 'VECTORED' trained parameters(follow the course notes),
        include W1 [hidden_size, visible_size],
                W2 [visible_size, hidden_size],
                b1 [hidden_size],
                b2 [visible_size]
        example code for getting W1, W2, b1, b2 from theta:
            W1 = theta[0: hidden_size * visible_size].reshape(hidden_size, visible_size)
            b1 = theta[2 * hidden_size * visible_size: 2 * hidden_size * visible_size + hidden_size]
            W2 = theta[hidden_size * visible_size: 2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
            b2 = theta[2 * hidden_size * visible_size + hidden_size:]
    """
    """STEP 0: Here we provide the relevant parameters values that will
    allow your sparse autoencoder to get good filters; you do not need to
    change the parameters below."""
    # visible_size = 28 * 28
    # hidden_size = 196
    # sparsity_param = 0.1
    # lambda_ = 3e-3
    # beta = 3

    """STEP 1: Implement sampleIMAGES
    After implementing sampleIMAGES, the display_network command should
    display a random sample of 200 patches from the dataset"""
    # patches = sample_images.sample_images() # Loading Sample Images
    # Loading 10K images from MNIST database
    patches = input_data
    theta = initialize(hidden_size, visible_size)  # Obtain random parameters theta

    """STEP 2: Implement sparseAutoencoderCost
    You can implement all of the components (squared error cost, weight decay term,
    sparsity penalty) in the cost function at once, but it may be easier to do
    it step-by-step and run gradient checking (see STEP 3) after each step.  We
    suggest implementing the sparseAutoencoderCost function using the following steps:
        (a) Implement forward propagation in your neural network, and implement the
            squared error term of the cost function.  Implement backpropagation to
            compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking
            to verify that the calculations corresponding to the squared error cost
            term are correct.
        (b) Add in the weight decay term (in both the cost function and the derivative
            calculations), then re-run Gradient Checking to verify correctness.
        (c) Add in the sparsity penalty term, then re-run Gradient Checking to
            verify correctness.
    Feel free to change the training settings when debugging your
    code.  (For example, reducing the training set size or
    number of hidden units may make your code run faster; and setting beta
    and/or lambda to zero may be helpful for debugging.)  However, in your
    final submission of the visualized weights, please use parameters we
    gave in Step 0 above."""
    cost, grad = sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, patches)

    """STEP 3: Gradient Checking
    Hint: If you are debugging your code, performing gradient checking on smaller models
    and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
    units) may speed things up.
    First, lets make sure your numerical gradient computation is correct for a
    simple function.  After you have implemented computeNumericalGradient.m,
    run the following:"""
    if debug:
        gradient.check_gradient()
        # Now we can use it to check your cost function and derivative calculations for the sparse autoencoder.
        # J is the cost function
        J = lambda x: sparse_autoencoder_cost(x, visible_size, hidden_size, lambda_, sparsity_param, beta, patches)
        num_grad = gradient.compute_gradient(J, theta)
        # Use this to visually compare the gradients side by side
        print(num_grad, grad, sep='\n')
        # Compare numerically computed gradients with the ones obtained from backpropagation
        diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
        print(diff)
        print("Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n\n")

    """STEP 4: After verifying that your implementation of
    sparseAutoencoderCost is correct, You can start training your sparse
    autoencoder with minFunc (L-BFGS)."""
    theta = initialize(hidden_size, visible_size)  # Randomly initialize the parameters
    J = lambda x: sparse_autoencoder_cost(x, visible_size, hidden_size, lambda_, sparsity_param, beta, patches)
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options={
        'maxiter': 400,
        'disp': True
    })
    opt_theta = result.x

    """STEP 5: Visualization"""
    # display_network.display_network(W1)
    return opt_theta
