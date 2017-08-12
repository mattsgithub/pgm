import pytest

from pgm.graph import DirectedGraph


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
