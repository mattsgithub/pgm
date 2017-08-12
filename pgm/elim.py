import copy


def get_min_fill_cost(R, n):
    """If n were eliminated,
    how many fill-ins would this introduce?
    """
    cost = 0

    # List of neighbors
    N = list(R[n])
    for i in xrange(len(N)):
        for j in xrange(i + 1, len(N)):
            if N[j] not in R[N[i]]:
                cost += 1
    return cost


def get_elim_order_using_greedy_search(G):
    """Greedy search algorithm
    Uses min fill as cost function
    Modifies G
    See triangulation-report
    """
    I = copy.deepcopy(G)
    R = copy.deepcopy(G)
    elim_order = []

    while len(R) > 0:
        min_cost = float('inf')
        opt_node = None

        for n in R:
            cost = get_min_fill_cost(R, n)
            if cost < min_cost:
                min_cost = cost
                opt_node = n

        elim_order.append(opt_node)

        for n1 in R[opt_node]:
            for n2 in R[opt_node]:
                if n1 != n2:
                    I[n1].add(n2)
                    R[n1].add(n2)

        # Remove opt_node from induced graph
        del R[opt_node]
        for n in R:
            if opt_node in R[n]:
                R[n].remove(opt_node)

    return I, elim_order
