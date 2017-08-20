from collections import defaultdict
import numpy as np

from pgm.factor import Factor
from pgm.elim import get_elim_order
import networkx as nx


def get_clique_graph(I, elim_order):
    """Construct clique graph
       from induced graph and given
       elim order. If elim order is
       followed, the neighbors of
       each node (including itself)
       is guaranteed to form a clique
    """
    C = nx.Graph()
    return C


class BayesianNetwork(object):
    def __init__(self):
        self._nodes = defaultdict(dict)
        self._index = 0
        self._dg = nx.DiGraph()
        self._to_index = dict()

    def load_graph_from_json(self, json_):
        self._dg = nx.DiGraph()
        for node_name, value in json_.iteritems():

            # Have we seen this node name before?
            if node_name not in self._to_index:
                self._to_index[node_name] = self._index
                self._index += 1

            for p in value['parents']:
                if p not in self._to_index:
                    self._to_index[p] = self._index
                    self._index += 1

            p_indx = [self._to_index[p] for p in value['parents']]
            c_indx = self._to_index[node_name]

            scope = np.array([c_indx] + p_indx)
            card = np.ones(scope.shape[0], dtype=np.int32) * 2
            val = np.array([v['value'] for v in value['values']])

            # What are the parents?
            f = Factor(scope=scope,
                       card=card,
                       val=val)
            self.add_node(node_name=node_name,
                          attr_dict={'factor': f})
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
        ug = self._dg.to_undirected()
        I, elim_order = get_elim_order(ug)
        # Form a clique graph now that
        # induced graph and elim order
        # is known. This clique graph
        # will be a tree. Thus, belief
        # propagation can be computed on
        # it
        c = get_clique_graph(I, elim_order)

        self.set_value('parent', 'rv', np.array([0.57, 0.43]))
        self.set_value('child', 'rv', np.array([0.284, 0.716]))

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
