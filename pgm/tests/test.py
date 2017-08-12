import networkx as nx
import itertools
from itertools import combinations

from pgm.prob import DiscreteVariable
from pgm.factor import get_df_from_variables
from pgm.factor import Factor


def get_clique(G, var):
    c = None
    for name in G.node:
        node = G.node[name]
        if node['type'] == 'clique' and var in node['factor'].scope:
            c = name
            break
    return c


def infer(G, c_name, var):
    # Find
    f = G.node[c_name]['posterior']
    for d in f.scope:
        if d != var:
            f = f.marginalize(d)
    print f


def get_clique_leaf_to_separator_msg(node):
    return node['factor']


def get_clique_to_separator_msg(parent_node, node):
    # This factor is not update!
    msg = node['factor']
    for m in node['messages']:
        msg = msg * m
    # Now, marginalize except for
    # parent scope
    if parent_node is not None:
        S_keep = parent_node['factor'].scope
        S_remove = msg.scope
        for c in S_remove:
            if c in S_keep:
                continue
            msg = msg.marginalize(c)
    return msg


def get_separator_to_clique_msg(node):
    msg = node['factor']
    for m in node['messages']:
        msg = msg * m
    return msg


def propagate_up(G, node_name, parent_node_name, is_leaf=False):
    parent_node = None
    msg = None

    if parent_node_name in G.node:
        parent_node = G.node[parent_node_name]
    node = G.node[node_name]

    if is_leaf and node['type'] == 'clique':
        msg = get_clique_leaf_to_separator_msg(node)
    elif node['type'] == 'separator':
        msg = get_separator_to_clique_msg(node)
    elif node['type'] == 'clique':
        msg = get_clique_to_separator_msg(parent_node, node)
    else:
        raise Exception('Unknown Type')

    if parent_node is None:
        # Reached end
        node['posterior'] = msg
        return

    parent_node['messages'].append(msg)
    num_msg = len(parent_node['messages'])
    size = parent_node['mailbox_size']

    if size == num_msg:
        _node_name = parent_node_name
        _parent_node_name = parent_node['parent']
        propagate_up(G, _node_name, _parent_node_name)


def propagate_down(G, node_name, parent_node_name=None):
    ns = set(G.neighbors(node_name))

    # Remove from set
    if parent_node_name is not None:
        ns.remove(parent_node_name)

    nx.set_node_attributes(G, 'mailbox_size', {node_name: len(ns)})
    nx.set_node_attributes(G, 'messages', {node_name: list()})

    for c in ns:
        nx.set_node_attributes(G, 'parent', {c: node_name})
        propagate_down(G, c, node_name)

    # Leaf node
    if len(ns) == 0:
        propagate_up(G, node_name, parent_node_name, True)


def compile(G, c_name):
    propagate_down(G, c_name)


def clique_graph(G):
    # This will be the clique graph
    H = nx.Graph()

    # Create clique nodes
    cliques = list(nx.find_cliques(G))

    # Create new factor from each clique
    for i, clique in enumerate(cliques):
        f = reduce(lambda a, b: a*b, [G.node[n]['factor'] for n in clique])
        H.add_node('C{0}'.format(i), parent=None, type='clique', factor=f)

    # Add edges and weight them
    for c1_name, c2_name in combinations(H.node.keys(), 2):
        var1 = H.node[c1_name]['factor'].scope
        var2 = H.node[c2_name]['factor'].scope
        common_vars = var1.intersection(var2)

        size_ = len(common_vars)
        if len(common_vars) > 0:
            H.add_edge(c1_name, c2_name, weight=-size_)
    return H


def dedupe_clique_graph(G):
    # Select subset of graph
    H = nx.minimum_spanning_tree(G, weight='weight')
    return H


def clique_graph_with_separator_nodes(G):
    # Now, create separator nodes
    s = 0
    H = G.copy()
    for c1_name, c2_name in combinations(H.node.keys(), 2):
        # Do they form a connection?
        if not H.has_edge(c1_name, c2_name):
            continue
        # They are connected
        var1 = H.node[c1_name]['factor'].scope
        var2 = H.node[c2_name]['factor'].scope
        common_vars = var1.intersection(var2)
        V = []
        for v in common_vars:
            V.append(DiscreteVariable(v))
        factor = Factor(get_df_from_variables(V, default_value=1.))

        s_name = 'S{0}'.format(s)
        s += 1
        H.add_node(s_name,
                   type='separator',
                   parent=None,
                   factor=factor)

        H.remove_edge(c1_name, c2_name)
        H.add_edge(c1_name, s_name)
        H.add_edge(s_name, c2_name)
    return H


def draw(G):
    from matplotlib import pyplot as plt
    pos = nx.fruchterman_reingold_layout(G)

    dict_ = {k: '{0}: {1}'.format(k, G.node[k]['factor'].scope) for k in G.node}
    nx.draw_networkx_labels(G, pos, labels=dict_)
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=G.nodes(),
                           node_color='w',
                           node_size=500,
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos)
    plt.show()


def triangulate(G, elim_order=None):
    H = G.copy()
    I = G.copy()

    # If elim order is None
    # Just use order of keys
    if elim_order is None:
        # elim_order = H.nodes()
        # random.shuffle(elim_order)
        elim_order = ['P2', 'D', 'E2', 'F', 'P3', 'C', 'E1', 'P1']

    for node_name in elim_order:
        # Get neighbors
        neighbors = H.neighbors(node_name)

        # Now, remove this node
        H.remove_node(node_name)

        # Now, connect all neighbors
        for n1, n2 in combinations(neighbors, 2):
            I.add_edge(n1, n2)
    return I


def moralize(G):
    '''Returns moralized graph'''
    H = G.to_undirected()
    for node in G.nodes():
        parents = G.predecessors(node)
        for n1, n2 in combinations(parents, 2):
            H.add_edge(n1, n2)
    return H


def get_factor(variable, variables, entries):
    f = Factor(get_df_from_variables(variables))
    for e in entries:
        f.set_value(e[0], e[1])
    return f


def get_summary_or_dist(parents, child):
    L = []
    entries = []
    for p in parents:
        L.append(((p, 'true'), (p, 'false')))
    L.append(((child, 'true'), (child, 'false')))
    for element in itertools.product(*L):
        p = 1.0 if 'true' in [e[1] for e in element[:-1]] else 0.0
        if element[-1][1] == 'false':
            p = 1. - p
        entries.append((element, p))
    return tuple(entries)

bn = nx.DiGraph()

P1 = DiscreteVariable('P1')
P2 = DiscreteVariable('P2')
P3 = DiscreteVariable('P3')
C = DiscreteVariable('C')
D = DiscreteVariable('D')
E1 = DiscreteVariable('E1')
E2 = DiscreteVariable('E2')
F = DiscreteVariable('F')

# Defines topology
bn.add_node('P1')
bn.add_node('P2')
bn.add_node('P3')
bn.add_node('C')
bn.add_node('D')
bn.add_node('E1')
bn.add_node('E2')
bn.add_node('F')
bn.add_edge('P1', 'C')
bn.add_edge('P2', 'C')
bn.add_edge('P3', 'C')
bn.add_edge('C', 'D')
bn.add_edge('D', 'E1')
bn.add_edge('D', 'E2')
bn.add_edge('E1', 'F')
bn.add_edge('E2', 'F')

entries = (
               ((('P1', 'true'),), 0.7),
               ((('P1', 'false'),), 0.3)
              )
f = get_factor(P1, (P1,), entries)
bn.node['P1']['factor'] = f


entries = (
               ((('P2', 'true'),), 0.7),
               ((('P2', 'false'),), 0.3)
              )
f = get_factor(P2, (P2,), entries)
bn.node['P2']['factor'] = f

entries = (
               ((('P3', 'true'),), 0.7),
               ((('P3', 'false'),), 0.3)
              )
f = get_factor(P3, (P3,), entries)
bn.node['P3']['factor'] = f

# Set CPTs
dist = get_summary_or_dist(['P1', 'P2', 'P3'], 'C')
f = get_factor(C, (C, P1, P2, P3), dist)
bn.node['C']['factor'] = f

dist = get_summary_or_dist(['C'], 'D')
f = get_factor(D, (C, D), dist)
bn.node['D']['factor'] = f


dist = get_summary_or_dist(['D'], 'E1')
f = get_factor(E1, (E1, D), dist)
bn.node['E1']['factor'] = f


dist = get_summary_or_dist(['D'], 'E2')
f = get_factor(E2, (E2, D), dist)
bn.node['E2']['factor'] = f


dist = get_summary_or_dist(['E1', 'E2'], 'F')
f = get_factor(F, (E1, E2, F), dist)
bn.node['F']['factor'] = f


G = moralize(bn)
G = triangulate(G)
G = clique_graph(G)
G = dedupe_clique_graph(G)
G = clique_graph_with_separator_nodes(G)
name = 'C'
c_name = get_clique(G, name)
compile(G, c_name)
infer(G, c_name, name)

'''
other [97.3, 2.70]
Pi  = [70, 30]
'''
