# Assignment 1, Part 3: Nearest Neighbor Classification
#

import math
import random
from assignment_one_kmeans import read_data, \
    num_unique_labels, euclidean_squared


def get_fold_indices(n, num_folds, fold_id):
    """ Returns training indices and test indices for a given fold_id.

    n: number of samples (so indices are in range(n))
    num_folds: number of total folds
    fold_id: the id of the fold we are creating

    Let fold_size = integer part of n/num_folds.
    If fold_id is i (0 <= i < num_folds-1) we return these indices:

    [0, 1, ..., i*fold_size-1] + [(i+1)*fold_size, ..., n-1]

    as the training indices. If fold_id is num_folds-1 (last fold)
    we return these indices are training indices:

    [0, 1, ..., (num_folds-1)*fold_size-1]

    as the training indices.

    In all cases, the complement (relative to range(n))
    of the training indices gives the testing indices.
    """

    # check whether fold_id is in the right range
    if fold_id < 0 or fold_id >= num_folds:
        msg = 'Fold id %d illegal for supplied number of folds (which is %d)' \
            % (fold_id, num_folds)
        raise Exception(msg)

    # n and num_folds are ints so this
    # returns the integer part of the division
    fold_size = n/num_folds

    # last fold_id has to be treated slightly differently:
    # when n is not a multiple of num_folds, the training
    # indices set for the last fold will be a bit smaller
    # (and hence the test indices set will be a bit larger)
    if fold_id == num_folds-1:
        # TASK 3.1.1
        # set train_indices correctly
        train_indices = range(0, 0)
    else:
        # TASK 3.1.2
        # set train_indices correctly
        # by concatenating two ranges
        train_indices = range(0, 0) + range(0, 0)

    # TASK 3.1.3
    # set test_indices to those indices in range(n)
    # that are not in train_indices
    test_indices = []

    return train_indices, test_indices


def nn_classifier(point, train_data, train_labels, k, K):
    """ Implements k-nearest neighbors classification rule.

    Finds k nearest neighbors of point in train_data. Then
    the most frequently appearing train_labels (among the labels
    of these neighbors) are calculated and one of them is
    returned at random.

    K is the number of possible labels.
    (Possible labels are assumed to be 0, 1, ..., K-1.)
    """

    # compute distances of training examples to this point
    distances = [math.sqrt(euclidean_squared(x, point)) for x in train_data]

    # put distances, taining examples and training labels together
    all_three = zip(distances, train_data, train_labels)

    # TASK 3.2.1
    # sort the triples by distances
    all_three_sorted = []

    # TASK 3.2.2
    # after sorting extract the labels from the first k
    # triples
    nearest_k_labels = []

    # initialize label frequencies to 0's
    freq = [0]*K

    # TASK 3.2.3
    # compute labels frequencies in nearest_k_labels
    for label in range(K):
        freq[label] = len([])

    # find the labels that have maximum frequency
    max_freq = max(freq)
    max_freq_labels = [x for x in range(K) if freq[x] == max_freq]

    # if there are several labels with max frequency,
    # return one of them at random
    return random.choice(max_freq_labels)


def classification_error(classifier, data, labels):

    """ Compute classification error of classifier on data & labels. """

    # check whether data and labels have same no. of elements
    if len(data) != len(labels):
        raise Exception('Unequal number of data and labels.')
    else:
        n = len(data)

    # compute list of booleans containing True for examples
    # classified incorrectly and False for examples
    # classified correctly
    error_indicators = [classifier(data[i]) != labels[i] for i in range(n)]

    # TASK 3.3
    # use the list of booleans above to calculate total number of errors
    total_error = 0

    # don't want integer division to occur, so convert total
    # error to float before dividing by n
    return float(total_error)/n


def main():

    data_file = 'seeds_dataset_shuffled.txt'
    instances, labels = read_data(data_file)

    # want labels run from 0 through K-1
    # not 1 through K
    labels = [i-1 for i in labels]

    print 'Read %d instances and %d labels from file %s.' \
        % (len(instances), len(labels), data_file)

    if len(instances) != len(labels):
        raise Exception('Expected equal number of instances and labels.')
    else:
        n = len(instances)

    # Find number of clusters by finding out how many unique elements are there
    # in labels.
    K = num_unique_labels(labels)
    print 'Found %d unique labels.' % K

    # k-nearest neighbor classification for various k
    k_range = range(1, 31)

    # create empty list to store cross-validation errors for different k
    cv_error = []

    # 10-fold cross-validation
    num_folds = 10

    for k in k_range:

        total_error = 0.0

        # create and process all folds
        for fold_id in range(num_folds):

            # separate indices into training indices
            # and test indices for this particular fold
            fold_train_indices, fold_test_indices = \
                get_fold_indices(n, num_folds, fold_id)

            # TASK 3.4.1
            # get the training data and labels
            # and create a k-NN classifier
            train_data = []
            train_labels = []
            classifier = None

            # TASK 3.4.2
            # get the test data and labels
            # and evaluate the classifier's error
            test_data = []
            test_label = []
            fold_error = classification_error(classifier,
                                              test_data, test_label)

            total_error += fold_error

        cv_error.append(total_error/num_folds)

    # print the values for k and the corresponding cross-validation errors
    for i in range(len(k_range)):
        print k_range[i], cv_error[i]


if __name__ == '__main__':
    main()
