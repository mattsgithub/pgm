from collections import defaultdict


class DirectedGraph(object):
    def __init__(self):
        self._nodes = defaultdict(dict)

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
