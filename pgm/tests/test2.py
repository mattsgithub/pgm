import numpy as np

from pgm.factor import Factor
from pgm.factor import get_product_from_list
from pgm.factor import get_marg
from pgm.graph import get_directed_adjacency_dict
from pgm.graph import get_moralized_adjacency_dict
from pgm.graph import get_clique_graph
from pgm.elim import get_elim_order_using_greedy_search


f1 = Factor(scope=np.array([0]),
            card=np.array([2]),
            val=np.array([0.7, 0.3]))

f2 = Factor(scope=np.array([1]),
            card=np.array([2]),
            val=np.array([0.7, 0.3]))

f3 = Factor(scope=np.array([2]),
            card=np.array([2]),
            val=np.array([0.1, 0.9]))

f4 = Factor(scope=np.array([3, 0, 1, 2]),
            card=np.array([2, 2, 2, 2]),
            # Deterministic OR Distribution
            val=np.array([1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          1.0]))

f5 = Factor(scope=np.array([4, 3]),
            card=np.array([2, 2]),
            val=np.array([0.8, 0.2, 0.1, 0.9]))

f6 = Factor(scope=np.array([5, 4]),
            card=np.array([2, 2]),
            val=np.array([0.45, 0.55, 0.23, 0.77]))

f7 = Factor(scope=np.array([6, 4]),
            card=np.array([2, 2]),
            val=np.array([0.11, 0.89, 0.75, 0.25]))

f8 = Factor(scope=np.array([7, 5, 6]),
            card=np.array([2, 2, 2]),
            val=np.array([1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          0.0,
                          1.0,
                          1.0]))

factors = [f1, f2, f3, f4, f5, f6, f7, f8]
posteriors = np.array([0.7, 0.7, 0.1, 0.919, 0.7433, 0.393526, 0.274288, 0.5867399])


def _get_clique_dag(C, parent, C_, visited):
    visited.add(parent)
    if len(visited) == len(C):
        return C_

    children = [c for c in C[parent] if c not in visited]
    for c in children:
        C_[parent]['children'].add(c)
        C_[c]['parents'].add(parent)
        return _get_clique_dag(C, c, C_, visited)


def get_clique_dag(C):
    C_ = [{'parents': set(), 'children': set(), 'mailbox': []} for _ in xrange(len(C))]
    visited = set()
    return _get_clique_dag(C, 0, C_, visited)


def msg_pass(C, cliques, factors, i, msg):
    C[i]['mailbox'].append(msg)

    # Is the mailbox full?
    if len(C[i]['mailbox']) == len(C[i]['children']):

        # Create factor
        fs = [factors[k] for k in cliques[i].factors] + C[i]['mailbox']
        f = get_product_from_list(fs)
        if len(C[i]['parents']) > 0:
            for j in C[i]['parents']:
                v = cliques[i].vars
                r = cliques[j].vars
                vars_ = v.difference(r)
                for v in vars_:
                    f = get_marg(f, v)
                msg_pass(C, cliques, factors, j, f)
        else:
            C[i]['mailbox'] = [f]


def run_inference(C, cliques, factors):
    # Find children
    children = [i for i in range(len(C)) if len(C[i]['children']) == 0]
    for i in children:
        fs = [factors[k] for k in cliques[i].factors]
        f = get_product_from_list(fs)
        # Send message to each of their parents
        for j in C[i]['parents']:
            # Child vars
            v = cliques[i].vars
            # Parent vars
            r = cliques[j].vars
            vars_ = v.difference(r)
            for v in vars_:
                f = get_marg(f, v)
            msg_pass(C, cliques, factors, j, f)


G = get_directed_adjacency_dict(factors)
M = get_moralized_adjacency_dict(G)

# Find elimorder and induced graph
I, elim_order = get_elim_order_using_greedy_search(M)

# Undirected graph
C, cliques = get_clique_graph(I, elim_order)

# Convert to DAG
C = get_clique_dag(C)
run_inference(C, cliques, factors)

f = C[0]['mailbox'][0]
print f
f = get_marg(f, 2)
f = get_marg(f, 0)
f = get_marg(f, 3)
print f
N = np.sum(f.val)
print f.val * (1./N)
