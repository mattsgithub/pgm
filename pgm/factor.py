from collections import namedtuple
import numpy as np


Factor = namedtuple('Factor', 'scope card val')


def assign_to_indx(A, C):
    """Given assignment(s) and
       card of factor, returns
       index/indices
     """

    # Returns M x 1
    # return A.dot(s.reshape((len(s), 1)))
    # Returns 1 dimesional array
    s = get_stride(C)
    return A.dot(s.reshape((len(s), 1))).reshape((A.shape[0],))


def get_stride(card):
    '''Returns 1-dimesional numpy array
       First element, by definition, is always
       1. Because it's the fastest moving
       column
    '''
    return np.cumprod(np.concatenate((np.array([1]), card[:-1])))


def indx_to_assign(i, c):
    """Calculates assignment vectors for
    a given index vector, i

    i: 1d np.array
    c: 1d np.array

    Idea is to take vector i and create matrix:

    i0 i0 i0
    i1 i1 i1

    Cast stride vector as column vector:
    s0
    s1
    s2


    I / s gives...

    i0/s0  i0/s1  i0/s2
    i1/s0  i1/s1  i1/s2

    No need to take floor, because we
    are dividing integers...

    Finally, take mod car

    x = floor(index / phi.stride(i))
    assignment[i] = x  mod card[i]
    """
    s = get_stride(c)
    i_ = i.reshape((len(i), 1))
    I = np.repeat(i_, len(s), axis=1)
    return np.mod(I / s, c)


def get_marg(A, v):
    # Get indices for every variable
    # except one being summed out
    map_ = np.where(A.scope != v)

    # Scope of returning factor
    scope = A.scope[map_]

    # Card of returning factor
    card = A.card[map_]

    # Init val of returning factor
    N = np.prod(card)
    val = np.zeros((N,))

    # Indices for all vals in A
    I = np.arange(0, len(A.val))

    # Get complete assignment matrix for A
    A_ = indx_to_assign(I, A.card)

    indx = assign_to_indx(A_[:, map_], card)

    # TODO: Redo without for loop?
    # For same index values, sum, and place
    # at index for returning factor
    for i, j in enumerate(indx):
        # Corresponding
        v = A.val[i]
        val[j] += v

    return Factor(scope,
                  card,
                  val)


def get_product_from_list(fs):
    """Get product from a list of factors
    """
    if len(fs) == 0:
        raise ValueError('factor list must be greater than 1')

    if len(fs) == 1:
        return fs[0]

    for i in range(len(fs)):
        for j in range(len(fs)):
            if i != j:
                s = set(fs[i].scope)
                r = set(fs[j].scope)

                if len(s.intersection(r)) > 0:
                    f = get_product(fs[i], fs[j])
                    del fs[i]
                    del fs[j-1]
                    fs.append(f)
                    return get_product_from_list(fs)


def get_product(A, B):
    scope = np.union1d(A.scope, B.scope)

    # https://stackoverflow.com/questions/30684563/numpy-search-array-for-multiple-values-and-returns-their-indices
    # Index values that
    # respect ascending order
    # of elements
    indx = scope.argsort()

    # Sorted scope
    sorted_scope = scope[indx]
    mapA = indx[np.searchsorted(sorted_scope, A.scope)]
    mapB = indx[np.searchsorted(sorted_scope, B.scope)]

    card = np.empty((scope.shape[0],), dtype=int)

    np.put(card, mapA, A.card)
    np.put(card, mapB, B.card)

    N = np.prod(card)
    val = np.empty((N,))
    assign = indx_to_assign(np.arange(N), card)

    indx_A = assign_to_indx(assign[:, mapA], A.card)
    indx_B = assign_to_indx(assign[:, mapB], B.card)

    val = A.val[indx_A] * B.val[indx_B]

    return Factor(scope,
                  card,
                  val)
