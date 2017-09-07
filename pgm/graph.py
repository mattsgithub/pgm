from collections import defaultdict
import numpy as np
from itertools import combinations
import networkx as nx
from scipy.misc import comb

from pgm.factor import Factor
from pgm.factor import get_product_from_list
from pgm.factor import get_marg
from pgm.factor import assign_to_indx


def check_calibration(cg, to_index, to_name):
    beliefs = defaultdict(list)
    for clique_node, data in cg.nodes_iter(data=True):
        indices = data['belief'].scope
        node_names = [to_name[i] for i in indices]

        for i in xrange(len(node_names)):
            factor = data['belief']
            for j in xrange(len(node_names)):
                if i == j:
                    continue
                factor = get_marg(factor, to_index[node_names[j]])
            # Renormalize
            val = factor.val / factor.val.sum()
            beliefs[node_names[i]].append(val)
    for k, v in beliefs.iteritems():
        print k, v
        print ''


def message_pass(cg, to_name):
    root_node = cg.nodes()[0]
    upward_pass(root_node, cg, to_name)
    downward_pass(root_node, cg, to_name)


def compute_beliefs(cg):
    for n, dict_ in cg.nodes_iter(data=True):
        belief = [dict_['factor']]

        for m in dict_['msg_upward']:
            belief.append(m['msg'])

        for m in dict_['msg_downward']:
            belief.append(m['msg'])

        belief = get_product_from_list(belief)
        cg.node[n]['belief'] = belief


def upward_pass(root_node, cg, to_name):
    children = set(cg.neighbors(root_node))
    cg.node[root_node]['children'] = children
    for c in children:
        cg.node[c]['parent'] = root_node
        descend_first_pass(c, cg, to_name)


def downward_pass(root_node, cg, to_name):
    descend_second_pass(root_node, cg, to_name)


def get_msg(from_node, to_node, is_upward, cg, to_name):
    msg_type = 'msg_upward' if is_upward else 'msg_downward'

    # Factor for this node
    factors = [cg.node[from_node]['factor']]

    is_root = cg.node[from_node]['parent'] is None

    if is_root and 'msg_downard':
        for d in cg.node[from_node]['msg_upward']:
            if d['name'] == '{0}->{1}'.format(to_node, from_node):
                continue
            factors.append(d['msg'])
    elif len(cg.node[from_node][msg_type]) > 0:
        for d in cg.node[from_node][msg_type]:
            factors.append(d['msg'])
    msg = get_product_from_list(factors)

    # Now, marginalize out everything except sepset
    sepset = cg.get_edge_data(from_node, to_node)['sepset']
    scope = msg.scope
    for i in scope:
        name = to_name[i]
        if name not in sepset:
            msg = get_marg(msg, i)
    return msg


def ascend(node, parent_node, cg, to_name):
    # print 'ascend {0} --> {1}'.format(node, parent_node)
    msg = get_msg(from_node=node,
                  to_node=parent_node,
                  is_upward=True,
                  cg=cg,
                  to_name=to_name)

    msg_name = '{0}->{1}'.format(node, parent_node)
    cg.node[parent_node]['msg_upward'].append({'name': msg_name, 'msg': msg})

    # Is mailbox full?
    if len(cg.node[parent_node]['msg_upward']) == len(cg.node[parent_node]['children']):
        # If false, we've reached the root of the tree
        if cg.node[parent_node]['parent'] is not None:
            ascend(parent_node, cg.node[parent_node]['parent'], cg, to_name)


def descend_first_pass(node, cg, to_name):
    parent_node = cg.node[node]['parent']
    # print 'descend first pass {0} --> {1}'.format(parent_node, node)

    # Get children of node
    children = set(cg.neighbors(node))
    children.remove(parent_node)

    # Reached a leaf node
    if len(children) == 0:
        ascend(node, parent_node, cg, to_name)
    else:
        cg.node[node]['children'] = children
        for c in children:
            cg.node[c]['parent'] = node
            descend_first_pass(c, cg, to_name)


def descend_second_pass(parent_node, cg, to_name):
    # Send message to each child from parent
    for c in cg.node[parent_node]['children']:
        # print 'descend second pass {0} -> {1}'.format(parent_node, c)

        msg = get_msg(from_node=parent_node,
                      to_node=c,
                      is_upward=False,
                      cg=cg,
                      to_name=to_name)

        msg_name = '{0}->{1}'.format(parent_node, c)
        cg.node[c]['msg_downward'].append({'name': msg_name, 'msg': msg})

        descend_second_pass(c, cg, to_name)


def get_clique_graph(elim_order,
                     induced_graph,
                     to_name):

    clique_graph = nx.Graph()
    U = set()

    for node_name in elim_order:
        clique = set(induced_graph.neighbors(node_name))
        clique.add(node_name)

        clique_scope = set()
        factors = []

        for c in clique:
            if c not in U:
                factor = induced_graph.node[c]['factor']
                clique_scope.update(set([to_name[i] for i in factor.scope]))
                factors.append(factor)
        U.update(clique)
        if len(factors) == 0:
            continue

        factor = get_product_from_list(factors)

        attr_dict = {'scope': clique_scope,
                     'factor': factor,
                     'msg_upward': [],
                     'msg_downward': [],
                     'belief': None,
                     'children': [],
                     'parent': None}
        clique_name = 'C' + str(len(clique_graph) + 1)
        clique_graph.add_node(n=clique_name,
                              attr_dict=attr_dict)

    # Connect cliques that have any variables
    # in common
    for cn1, dict1 in clique_graph.nodes_iter(data=True):
        for cn2, dict2 in clique_graph.nodes_iter(data=True):
            if cn1 == cn2:
                continue
            sepset = dict1['scope'].intersection(dict2['scope'])
            n = len(sepset)
            if n > 0:
                clique_graph.add_edge(u=cn1,
                                      v=cn2,
                                      attr_dict={'sepset': sepset, 'weight': -n})

    clique_graph = nx.minimum_spanning_tree(clique_graph)
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
        cg = get_clique_graph(elim_order, induced_graph, self._to_name)

        if len(cg) > 1:
            message_pass(cg, self._to_name)
        compute_beliefs(cg)

        # check_calibration(cg, self._to_index, self._to_name)

        # Update random variables for all nodes in network
        for clique_node, data in cg.nodes_iter(data=True):

            indices = data['belief'].scope
            node_names = [self._to_name[i] for i in indices]

            for i in xrange(len(node_names)):
                factor = data['belief']
                for j in xrange(len(node_names)):
                    if i == j:
                        continue
                    factor = get_marg(factor, self._to_index[node_names[j]])
                # Renormalize
                val = factor.val / factor.val.sum()
                factor = Factor(scope=factor.scope,
                                card=factor.card,
                                val=val)
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
