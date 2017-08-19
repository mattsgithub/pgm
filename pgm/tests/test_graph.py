import pytest
import json

from pgm.graph import DirectedGraph
from pgm.factor import Factor
from pgm.factor import is_equal
import numpy as np


@pytest.fixture(scope='module')
def dgraph_gold():
    graph = {1: {'parents': set(), 'kv': {'key_1': 'value_1'}},
             2: {'parents': set(), 'kv': {'key_2': 'value_2'}},
             3: {'parents': {1, 2}, 'kv': {'key_3': 'value_3'}}}
    return graph


@pytest.fixture(scope='module')
def dgraph_test(dgraph_gold):
    # Create graph
    dg = DirectedGraph()
    for n, v in dgraph_gold.iteritems():
        dg.add_node(node_name=n,
                    dict_=v['kv'])
        dg.add_parents(node_name=n,
                       parent_names=v['parents'])
    return dg


def test_parents(dgraph_test, dgraph_gold):
    for n in dgraph_gold:
        parents = dgraph_test.get_parents(n)
        assert parents == dgraph_gold[n]['parents']


def test_dict_(dgraph_test, dgraph_gold):
    for n in dgraph_test.get_nodes():
        dict_ = dgraph_gold[n]['kv']
        for k, v in dict_.iteritems():
            assert v == dgraph_test.get_value(n, k)


def test_load_from_json():
    dg = DirectedGraph()
    with open('one_parent_net.json', 'r') as f:
        json_ = json.load(f)
        dg.load_graph_from_json(json_)

    # Check both nodes are present
    assert len(dg) == 2

    parents = dg.get_parents('parent')
    assert len(parents) == 0

    parents = dg.get_parents('child')
    assert len(parents) == 1

    factor = dg.get_value('parent', 'factor')
    f = Factor(scope=np.array([0]),
               card=np.array([2]),
               val=np.array([0.57, 0.43]))
    assert is_equal(f, factor)

    factor = dg.get_value('child', 'factor')
    f = Factor(scope=np.array([1, 0]),
               card=np.array([2, 2]),
               val=np.array([0.13, 0.6, 0.4, 0.87]))
    assert is_equal(f, factor)
