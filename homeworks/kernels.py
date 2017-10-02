import numpy as np

__all__ = ['linear', 'poly', 'rbf', 'sigmoid']


def linear(u, v):
    """ Returns inner product of u and v. """

    return np.inner(u, v)


def poly(degree, gamma, intercept=0.0):
    """ Returns polynomial kernel of specified degree and coeff gamma. """

    def poly_kernel_func(u, v):
        return (gamma*np.inner(u, v) + intercept) ** degree

    return poly_kernel_func


def rbf(gamma):
    """ Returns the gaussian/rbf kernel with specified gamma. """

    def rbf_kernel_func(u, v):
        return np.exp(-gamma*np.sum(np.abs(u - v) ** 2))

    return rbf_kernel_func


def sigmoid(gamma, intercept=0.0):
    """ Returns the sigmoid/tanh kernel with specified gamma. """

    def sigmoid_kernel_func(u, v):
        return np.arctan(gamma*np.inner(u, v) + intercept)

    return sigmoid_kernel_func


def logistic_loss_deriv(t, y):
    """ Returns derivative of logistic_loss(t, y) w.r.t. t. """

    return (-y * math.exp(-y*t)) / (1 + math.exp(-y*t))
