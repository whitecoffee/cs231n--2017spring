import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    f = np.dot(X[i], W)
    f -= np.max(f)  # Numeric stability
    exp_f = np.exp(f)
    exp_sum = np.sum(exp_f)
    
    loss -= np.log(exp_f[y[i]] / exp_sum)
    for j in range(num_classes):
        if j == y[i]:
            dW[:, j] -= (exp_sum - exp_f[y[i]]) * X[i] / exp_sum
        else:
            dW[:, j] += exp_f[j] * X[i] / exp_sum

  # Compute average
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  f = np.dot(X, W)
  f -= np.reshape(np.max(f, axis = 1), (-1, 1))
  exp_f = np.exp(f)
  exp_sum = np.sum(exp_f, axis = 1)
    
  loss = sum(-np.log( exp_f[np.arange(num_train), y] / exp_sum ))
  
  index = exp_f
  index[np.arange(num_train), y] = - (exp_sum - exp_f[np.arange(num_train), y])
  index /= np.reshape(exp_sum, (-1, 1))
  dW = np.dot(X.T, index)

  # Compute average
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

