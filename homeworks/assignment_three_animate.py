# Assignment 3, Part 2: Animate online algorithms

import numpy as np
import numpy.linalg as LA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from losses import logistic_loss_deriv


def generate_separable_data(n):
    """ Generate linearly-separable 2D data.

    Create n example, label pairs where examples are 2D points and labels are
    +1 or -1.
    """

    w0 = np.random.rand(2)  # choose true linear separator at random
    X = np.random.randn(n, 2)  # examples follow gaussian distribution
    Y = np.sign(X.dot(w0))  # select signs using true separator
    X = X + 0.1*np.outer(Y, w0)  # increase the margin a little bit

    return X, Y


# TASK 2.1
# implement this function
def perceptron_update(w, x, y):
    """ The perceptron update rule. """

    pass


def generate_nonseparable_data(n):
    """ Generate non linearly-separable 2D data.

    Create n example, label pairs where examples are 2D points and labels are
    +1 or -1. Labels are generated following a logistic model.
    """

    w0 = np.random.rand(2)  # choose true logistic parameter
    X = np.random.randn(n, 2)  # example follow gaussian distribution

    # compute probabilities using logistic transformation
    probs = np.exp(5*X.dot(w0))/(1+np.exp(5*X.dot(w0)))

    # generate Y[i] to be +1 with probability probs[i]
    # and -1 otherwise
    Y = 2*(np.random.rand(n) < probs)-1
    return X, Y


# TASK 2.2
# implement this function
def online_lr_update(w, x, y, stepsize=0.1):
    """ The online logistic regression update rule. """

    pass


def main():

    # declare w global so that all functions can use and modify it
    global w

    # algorithms we'll animate
    names = ['Perceptron', 'Online_logistic_regression']

    # data generation functions for the algorithms
    data_generators = [generate_separable_data, generate_nonseparable_data]

    # the update rules of the algorithms
    updaters = [perceptron_update, online_lr_update]

    # loop over algorithms to animate
    for name, generator, algo in zip(names, data_generators, updaters):

        print 'Running ' + name + ' algorithm...'

        # initialize to x-axis as the separator
        w = np.array([0, 1])

        n = 50  # sample size
        X, Y = generator(n)
        positives = (Y == 1)
        negatives = (Y == -1)

        # get values to plot for drawing the separator
        min_x1 = np.amin(X[:, 0])
        max_x1 = np.amax(X[:, 0])
        x1_vals = np.linspace(min_x1, max_x1, 100)
        x2_vals = -w[0]*x1_vals/w[1]

        # create new figure
        fig = plt.figure()

        # TASK 2.3.1
        # plot positives as blue dots
        line1, = plt.plot()

        # TASK 2.3.2
        # plot negatives as red dots
        line2, = plt.plot()

        # TASK 2.3.3
        # plot separator as a black line
        line3, = plt.plot()

        # current point is a yellow star
        line4, = plt.plot(X[0, 0], X[0, 1], '*y', markersize=10)

        def frame_update(i):
            """ Update frame for animating an algorithm update rule. """

            global w

            # don't do anything if it's the first frame
            if i == 0:
                return

            # else pick (i-1)th example (modulo sample size)
            i = (i-1) % n

            # TASK 2.4.1
            # execute online update
            w = w

            # TASK 2.4.2
            # draw updated separator
            x2_vals = x2_vals
            line3.set_ydata()

            # TASK 2.4.3
            # move the star to the point just processed
            line4.set_xdata()
            line4.set_ydata()

        # create animation
        # go through example twice
        # insert 200 milliseconds interval between frames
        ani = animation.FuncAnimation(fig, frame_update,
                                      range(2*n+1), interval=200)

        # save animation to gif file
        filename = '%s_anim.gif' % name
        print 'Saving ' + filename
        mpl.animation.ImageMagickBase.output_args = ['-loop', '1', filename]
        ani.save(filename, writer='imagemagick')

        # close the figure
        plt.close(fig)


if __name__ == '__main__':
    main()
