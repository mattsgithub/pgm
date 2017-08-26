from collections import defaultdict
import numpy as np
from itertools import combinations
import networkx as nx
from scipy.misc import comb

from pgm.factor import Factor
from pgm.factor import get_product_from_list
from pgm.factor import get_marg
from pgm.factor import assign_to_indx


def message_pass(cg, to_name):
    # Pick first node as root
    # Perform depth first search
    root_node = cg.nodes()[0]
    descend(root_node, None, cg, to_name)


def ascend(node, parent_node, cg, to_name):
    # Pass message from child to parent
    # Collect all messages
    factors = [cg.node[node]['factor']]
    if len(cg.node[node]['msg']) > 0:
        factors.extend(cg.node[node]['msg'])
    msg = get_product_from_list(factors)

    # Now, marginalize out everything except sepset
    sepset = cg.get_edge_data(node, parent_node)['sepset']
    scope = msg.scope
    for i in scope:
        name = to_name[i]
        if name not in sepset:
            msg = get_marg(msg, i)

    cg.node[parent_node]['msg'].append(msg)

    # Is mailbox full?
    if len(cg.node[parent_node]['msg']) == len(cg.node[parent_node]['children']):
        # Don't continue if parent is None;
        # Reached end
        if cg.node[parent_node]['parent'] is not None:
            ascend(cg.node[parent_node], cg.node[parent_node]['parent'], cg, to_name)


def descend(node, parent_node, cg, to_name):
    children = set(cg.neighbors(node))
    if parent_node is not None:
        children.remove(parent_node)

    # Reached a leaf node
    if len(children) == 0:
        ascend(node, parent_node, cg, to_name)
    else:
        if parent_node is not None:
            cg.node[parent_node]['children'] = children
        for c in children:
            cg.node[c]['parent'] = node
            descend(c, node, cg, to_name)


def get_clique_graph(elim_order,
                     induced_graph):

    clique_graph = nx.Graph()
    node_names_with_used_factors = set()

    for node_name in elim_order:

        # Establish scope for this clique
        clique_scope = set(induced_graph.neighbors(node_name))
        clique_scope.add(node_name)

        factors = []
        for clique_node_name in clique_scope:
            # Has this factor already been used?
            if clique_node_name in node_names_with_used_factors:
                continue

            # Can this factor map to this clique?
            if induced_graph.node[clique_node_name]['scope'] \
                    .issubset(clique_scope):
                factors.append(induced_graph.node[clique_node_name]['factor'])
                node_names_with_used_factors.add(clique_node_name)

        if len(factors) > 0:
            attr_dict = {'scope': clique_scope,
                         'factor': get_product_from_list(factors),
                         'msg': [],
                         'children': [],
                         'parent': None}
            clique_name = 'C' + str(len(clique_graph) + 1)
            clique_graph.add_node(n=clique_name,
                                  attr_dict=attr_dict)

            for cn, dict_ in clique_graph.nodes_iter(data=True):
                if cn == clique_name:
                    continue
                # TODO: This can lead non-trees
                sepset = clique_scope.intersection(dict_['scope'])
                if len(sepset) > 0:
                    clique_graph.add_edge(u=cn,
                                          v=clique_name,
                                          attr_dict={'sepset': sepset})

    return clique_graph


def get_elim_order(ug):
    induced_graph = ug.copy()
    elim_order = []
    unvisited_nodes = set(induced_graph.nodes(data=False))
    elim_order, induced_graph = _get_elim_order(induced_graph,
                                                unvisited_nodes,
                                                elim_order)
    return elim_order, induced_graph


def _get_elim_order(induced_graph,
                    unvisited_nodes,
                    elim_order):

    if len(unvisited_nodes) == 0:
        return elim_order, induced_graph

    elim_node = None
    global_min_fill_edges = float('inf')
    for node in unvisited_nodes:
        # Get neighboring nodes (exclude ones in elim_order)
        neighbors = set(induced_graph.neighbors(node)).difference(elim_order)

        # Edges among neighbors
        num_edges = 0
        for e1, e2 in combinations(neighbors, 2):
            if induced_graph.has_edge(e1, e2):
                num_edges += 1

        # And if they formed complete subgraph
        # how many edges then?
        max_num_edges = comb(len(neighbors), 2)

        # What is the difference?
        num_fill_edges = max_num_edges - num_edges

        # Is this less than the global minimum so far?
        if num_fill_edges < global_min_fill_edges:
            elim_node = node
            global_min_fill_edges = num_fill_edges

    # TODO: Is there a way to do this
    # within calls to networkx library?
    neighbors = set(induced_graph.neighbors(elim_node)).difference(elim_order)
    for n1, n2 in combinations(neighbors, 2):
        induced_graph.add_edge(n1, n2)

    # Add to elim order
    elim_order.append(elim_node)

    # Node is now visited
    unvisited_nodes.remove(elim_node)

    return _get_elim_order(induced_graph,
                           unvisited_nodes,
                           elim_order)

    return induced_graph, elim_order


def get_moral_graph(dg):
    ug = dg.to_undirected()
    for node in dg.nodes_iter(data=False):
        parents = dg.predecessors(node)
        for r in combinations(parents, 2):
            ug.add_edge(r[0], r[1])
    return ug


class BayesianNetwork(object):
    def __init__(self):
        self._nodes = defaultdict(dict)
        self._index = 0
        self._dg = nx.DiGraph()
        self._to_index = dict()
        self._to_name = dict()

    def load_graph_from_json(self, json_):
        self._dg = nx.DiGraph()
        for node_name, value in json_.iteritems():

            # Have we seen this node name before?
            if node_name not in self._to_index:
                self._to_index[node_name] = self._index
                self._to_name[self._index] = node_name
                self._index += 1

            for p in value['parents']:
                if p not in self._to_index:
                    self._to_index[p] = self._index
                    self._to_name[self._index] = p
                    self._index += 1

            p_indx = [self._to_index[p] for p in value['parents']]
            c_indx = self._to_index[node_name]

            scope = np.array([c_indx] + p_indx)
            card = np.ones(scope.shape[0], dtype=np.int32) * 2

            N = np.prod(card)
            val = np.empty((N,))
            for v in value['values']:
                A = np.array(v['states'])
                A = A.reshape((-1, A.shape[0]))
                i = assign_to_indx(A, card)[0]
                val[i] = v['value']

            # What are the parents?
            f = Factor(scope=scope,
                       card=card,
                       val=val)

            var_scope = set([node_name] + value['parents'])
            self.add_node(node_name=node_name,
                          attr_dict={'factor': f,
                                     'scope': var_scope})

            self.add_parents(node_name, value['parents'])

    def __len__(self):
        return len(self._dg)

    def add_parents(self,
                    child,
                    parent_names):
        for p in parent_names:
            self._dg.add_edge(p, child)

    def get_parents(self, node_name):
        return self._dg.predecessors(node_name)

    def get_nodes(self):
        for node_name in self._nodes:
            yield node_name

    def infer(self):
        ug = get_moral_graph(self._dg)
        elim_order, induced_graph = get_elim_order(ug)
        cg = get_clique_graph(elim_order, induced_graph)

        if len(cg) > 1:
            message_pass(cg, self._to_name)

        # Update random variables for all nodes in network
        for clique_node, data in cg.nodes_iter(data=True):
            node_names = list(data['scope'])

            for i in xrange(len(node_names)):
                factor = data['factor']
                for j in xrange(len(node_names)):
                    if i == j:
                        continue
                    factor = get_marg(factor, self._to_index[node_names[j]])
                self.set_value(node_names[i], 'rv', factor)

    def get_value(self,
                  node_name,
                  key):
        return self._dg.node[node_name][key]

    def set_value(self,
                  node_name,
                  key,
                  value):
        self._dg.node[node_name][key] = value

    def add_node(self,
                 node_name,
                 attr_dict=None):
        self._dg.add_node(n=node_name,
                          attr_dict=attr_dict)
