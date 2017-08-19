from collections import defaultdict
import numpy as np

from pgm.factor import Factor


class DirectedGraph(object):
    def __init__(self):
        self._nodes = defaultdict(dict)
        self._index = 0
        self._to_index = dict()

    def load_graph_from_json(self, json_):
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

            self.add_node(node_name, {'factor': f})
            self.add_parents(node_name, value['parents'])

    def __len__(self):
        return len(self._nodes)

    def add_parents(self,
                    node_name,
                    parent_names):
        self._nodes[node_name]['parents'].update(parent_names)

    def get_parents(self, node_name):
        return self._nodes[node_name]['parents']

    def add_key_values(self,
                       node_name,
                       dict_):
        self._nodes[node_name].update(dict_)

    def get_nodes(self):
        for node_name in self._nodes:
            yield node_name

    def get_value(self,
                  node_name,
                  key):
        return self._nodes[node_name][key]

    def add_node(self,
                 node_name,
                 dict_=None):
        self._nodes[node_name] = {'parents': set()}
        if dict_ is not None:
            self.add_key_values(node_name, dict_)
