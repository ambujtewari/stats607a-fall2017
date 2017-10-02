# Assignment 2, Part 1: Pagerank
#
# Version 0.1


import numpy as np
import numpy.linalg as LA


def compute_pageranks(google_mat, tol=1e-8):
    """ Compute the pagerank from the google matrix.

    Power method iterations are done until a tolerance of tol is achieved.
    """

    m, n = google_mat.shape
    if m != n:
        raise Exception('Expected first argument to be a square matrix')

    # TASK 1.5.1
    # start with the unif distribution
    dist = np.zeros(n)

    rel_change = 1.0
    while rel_change > tol:
        # TASK 1.5.2
        # perform the power method iteration
        new_dist = dist

        # compute relative change in iterate
        rel_change = LA.norm(new_dist - dist)/LA.norm(dist)
        dist = new_dist

    if not np.allclose(dist.dot(google_mat), dist):
        raise Exception('Power method did not find the stationary \
                        distribution')

    return dist


def main():

    indices_file = 'example_index.txt'
    edges_file = 'example_arcs.txt'

    # random surfing probability is 1-alpha
    alpha = 0.85

    # read webpage, index pairs from file
    page_indices = np.loadtxt(indices_file, dtype='|S30')
    n = len(page_indices)

    # forget the indices, they're just 0 through n-1
    page_names = page_indices[:, 0]

    # read the edges as pairs of integers
    # pairs have source first, destination second
    edges = np.loadtxt(edges_file, dtype='i8')

    # TASK 1.1
    # construct adjacency matrix with 1's exactly in
    # positions specified by row_id, col_id pairs in edges
    adj_mat = np.zeros((n, n))

    # construct 1d array of no. of outlinks
    out_links = np.sum(adj_mat, axis=1)

    # TASK 1.2
    # create 0-1 array with 1's for dangling nodes,
    # i.e. nodes without any out links.
    dangling = None

    # TASK 1.3
    # normalize non-zero rows, keep zero rows as is
    adj_mat_norm = np.zeros((n, n))

    # TASK 1.4
    # make matrix stochastic
    rank_one_change = np.zeros((n, n))
    adj_mat_stoch = adj_mat_norm + rank_one_change

    # rows of adj_mat_stoch should all sum to 1
    assert np.allclose(np.sum(adj_mat_stoch, axis=1), np.ones(n))

    # create the "google" matrix by adding a random surfing term
    google_mat = alpha*adj_mat_stoch + (1-alpha)*np.ones((n, n))/n

    # rows of adj_mat_stoch should all sum to 1
    assert np.allclose(np.sum(google_mat, axis=1), np.ones(n))

    pageranks = compute_pageranks(google_mat)

    # TASK 1.6
    # set top_10 to indices of top 10 pages with largest pageranks
    top_10 = None
    print page_names[top_10]

    # compute eigenvalues and eigenvectors
    w, V = LA.eig(google_mat.T)

    # TASK 1.7
    # use np.isclose() function and and numpy.ndarray.nonzero()
    # methods to find those indices where the eigenvalue vector
    # w has entries (numerically) close to 1
    # Make sure close_to_1 is a 1-d ndarray of indices
    close_to_1 = np.arange(n)

    if close_to_1.size:
        ind_of_1 = close_to_1[0]
    else:
        raise Exception('Expected at least 1 eigenvalue to be 1, none found')

    # extract the corresponding eignevector
    pageranks_eig = np.squeeze(V[:, ind_of_1])

    # TASK 1.8
    # eigenvector will be normalized to have euclidean norm 1
    # re-normalize it to be a probability distribution
    pageranks_eig = pageranks_eig

    # check whether our answer agrees with eig computation
    print 'Pagerank computations via power method and numpy.linalg.eig',
    if np.allclose(pageranks, pageranks_eig):
        print 'agree'
    else:
        print 'do not agree'


if __name__ == '__main__':
    main()
