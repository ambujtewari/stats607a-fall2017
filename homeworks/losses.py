import math

__all__ = ['squared_loss', 'squared_loss_deriv',
           'logistic_loss', 'logistic_loss_deriv']


def squared_loss(t, y):
    """ Returns (t-y) squared. """

    return (t-y)**2


def squared_loss_deriv(t, y):
    """ Returns derivative of (t-y)^2 w.r.t. t. """

    return 2*(t-y)


def logistic_loss(t, y):
    """ Return the logistic loss. """

    return math.log(1 + math.exp(-y*t))


def logistic_loss_deriv(t, y):
    """ Returns derivative of logistic_loss(t, y) w.r.t. t. """

    return (-y * math.exp(-y*t)) / (1 + math.exp(-y*t))
