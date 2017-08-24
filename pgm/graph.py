from collections import defaultdict
import numpy as np
from itertools import combinations
import networkx as nx

from pgm.factor import Factor
from pgm.factor import get_marg
from pgm.factor import get_product_from_list
from pgm.factor import assign_to_indx
from pgm.elim import get_elim_order


def run_message_passing(cg):
    if len(cg) == 1:
        return


def get_clique_graph(ug):
    # Return induced graph
    I, elim_order = get_elim_order(ug)
    index = 0
    clique_graph = nx.Graph()

    for node in elim_order:
        # Names of variables in clique
        node_names = set(I.neighbors(node))
        node_names.add(node)

        is_subset = False
        clique_neighbors = set()
        for clique_node, data in clique_graph.nodes_iter(data=True):
            if node_names.issubset(data['node_names']):
                is_subset = True
                break

            if len(node_names.intersection(data['node_names'])) > 0:
                clique_neighbors.add(clique_node)

        if not is_subset:
            clique_name = 'C{0}'.format(index)
            factor = get_product_from_list([ug.node[n]['factor'] for n in node_names])
            clique_graph.add_node(clique_name, {'node_names': node_names,
                                                'factor': factor})
            index += 1

            # Add neighbors
            for c in clique_neighbors:
                clique_graph.add_edge(clique_name, c)

        # Remove node from consideration
        I.remove_node(node)
    return clique_graph


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
        # Convert to moral graph
        ug = get_moral_graph(self._dg)

        # Convert to clique graph
        cg = get_clique_graph(ug)

        # cg = get_minimum_spanning_tree(cg)

        run_message_passing(cg)

        for clique_node, data in cg.nodes_iter(data=True):
            node_names = list(data['node_names'])

            for i in xrange(len(node_names)):
                factor = data['factor']
                # Unmarginalized factor
                for j in xrange(len(node_names)):
                    if i == j:
                        continue
                    import pdb; pdb.set_trace()
                    factor = get_marg(factor, self._to_index[node_names[j]])
                self.set_value(node_names[i], 'rv', factor)
        import pdb; pdb.set_trace()

        # self.set_value('parent', 'rv', np.array([0.57, 0.43]))
        # self.set_value('child', 'rv', np.array([0.284, 0.716]))

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
