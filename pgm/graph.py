from collections import namedtuple
from itertools import combinations
import copy

Clique = namedtuple('Clique', 'vars factors')


def get_clique_graph(I, elim_order):
    C = []
    cliques = []
    R = copy.deepcopy(I)
    return _get_clique_graph(R,
                             0,
                             elim_order,
                             C,
                             cliques)


def _get_clique_graph(R,
                      i,
                      elim_order,
                      C,
                      cliques):
    # We've reached the end
    # of all variables to examine
    if i == len(elim_order):
        return C, cliques

    # Names of variables involved
    # Forms a clique
    family = set([elim_order[i]])
    for j in R[elim_order[i]]:
        family.add(j)

    # Do we already have a clique
    # that subsumes this family?
    is_subset = False
    for c in cliques:
        if family.issubset(c.vars):
            is_subset = True
            break

    if not is_subset:
        used_variables = set()
        # TODO: Optimize; No need to loop every time
        # but works for now!
        for c in cliques:
            used_variables = used_variables.union(c.vars)
        factors = family.difference(used_variables)
        # Index of new clique created
        j = len(cliques)
        clique = Clique(vars=family, factors=factors)
        cliques.append(clique)
        C.append(set())

        # Establish neighbors of clique
        for k, c in enumerate(cliques[:-1]):
            # Should this clique be connected?
            if len(clique.vars.intersection(c.vars)) > 0:
                C[k].add(j)
                C[j].add(k)

    # Eliminate node from graph
    for n in R:
        if elim_order[i] in R[n]:
            R[n].remove(elim_order[i])
    del R[elim_order[i]]

    return _get_clique_graph(R,
                             i + 1,
                             elim_order,
                             C,
                             cliques)


def get_directed_adjacency_dict(factors):
    """Create dag given factors and scope
    Modeled as a list where each index is
    the node in the network.

    Key is node in network
    Value (set) is parents of node
    """
    G = {i: set() for i in range(len(factors))}
    for f in factors:
        # First element in scope is always
        # main variable
        i = f.scope[0]
        # Conditionals
        for p in f.scope[1:]:
            G[i].add(p)
    return G


def get_moralized_adjacency_dict(G):
    """Marry all parents
    """
    G = copy.deepcopy(G)
    for i in G:
        # Convert directed to undirected
        for parent in G[i]:
            G[parent].add(i)

        # Marry parents
        for pair in combinations(G[i], 2):
            G[pair[0]].add(pair[1])
            G[pair[1]].add(pair[0])
    return G
