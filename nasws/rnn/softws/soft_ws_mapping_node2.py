### DEFINE THE MAPPING FOR NODE 2 case
import logging
import random
from functools import partial

import utils
from copy import copy
from numpy import prod


from genotypes import PRIMITIVES, Genotype


def init_query(num_param_per_node, steps, init_v=0.0):
    query = []
    for i in range(steps):
        _q = [init_v,] * num_param_per_node[i]
        query.append(_q)
    return query


def map_v1(geno, geno_id, num_param_per_node, steps):
    """
    Map the linear structure.
    Activation functions

    """
    query = init_query(num_param_per_node, steps)

    a = PRIMITIVES.index(geno.recurrent[0][0]) - 1
    for i in range(steps):
        query[i][a] = 1.0
    return query


def map_v2(geno, geno_id, num_param_per_node, steps):
    """
    Map the linear structure.
    Activation functions

    """
    query = init_query(num_param_per_node, steps)

    a = PRIMITIVES.index(geno.recurrent[1][0]) - 1
    for i in range(steps):
        query[i][a] = 1.0
    return query


def map_v3(geno, geno_id, num_param_per_node, steps):
    """
    Map the linear structure.
    Activation functions

    """
    query = init_query(num_param_per_node, steps)

    a = PRIMITIVES.index(geno.recurrent[1][0]) - 1
    b = PRIMITIVES.index(geno.recurrent[0][0]) - 1
    query[0][a] = 1.0
    query[1][b] = 1.0

    return query


def _soft_map_v3(geno, geno_id, num_param_per_node, steps, init_v=0.1):
    """
    Map the linear structure.
    Activation functions

    """

    query = init_query(num_param_per_node, steps, init_v=init_v)
    other_v = 1. - init_v * 3
    a = PRIMITIVES.index(geno.recurrent[1][0]) - 1
    b = PRIMITIVES.index(geno.recurrent[0][0]) - 1
    query[0][a] = other_v
    query[1][b] = other_v

    return query


def map_random_v1(geno, geno_id, num_param_per_node, steps):
    # Pure random !!! increase the randomness.
    query = init_query(num_param_per_node, steps)
    for i in range(steps):
        query[i][ random.randint(0, num_param_per_node[i] - 1)] = 1.0
    return query


def soft_map_random_v1(geno, geno_id, num_param_per_node, steps):
    # Pure random !!! increase the randomness.
    query = init_query(num_param_per_node, steps, init_v=0.1)
    for i in range(steps):
        query[i][ random.randint(0, num_param_per_node[i] - 1)] = 0.7
    return query


class StoreQueryFull:
    id_query = {}

    def __init__(self, num_param_per_node, steps, search_space=None,**kwargs):
        self.num_param_per_node = num_param_per_node
        self.steps = steps
        self.search_space = search_space
        self._initialize_query()
        # logging.info("Stored Mapping of architecture: ")
        # logging.info(self.id_query)

    def _initialize_query(self):
        for i in range(self.search_space.num_solutions):
            self.id_query[i] = init_query(self.num_param_per_node, self.steps)

    def __call__(self, geno, geno_id, num_param_per_node, steps):
        return self.id_query[geno_id]


class _RandomQueryEql(StoreQueryFull):
    """
    Make sure the distribution is uniform distributed.
    each mapping has exactly 2 architectures.
    """
    def _initialize_query(self, init_v=0.0, v=1.0):
        archs_share = int(self.search_space.num_solutions / prod(self.num_param_per_node))
        queries = []
        for i in range(self.num_param_per_node[0]):
            for j in range(self.num_param_per_node[1]):
                q = init_query(self.num_param_per_node, self.steps, init_v=init_v)
                q[0][i] = v
                q[1][j] = v
                for k in range(archs_share):
                    queries.append(copy(q))

        assert len(queries) == self.search_space.num_solutions, 'Query number is not equal to num architectures.'
        geno_ids = [i for i in range(self.search_space.num_solutions)]
        random.shuffle(geno_ids)
        for i in geno_ids:
            self.id_query[i] = queries[i]


class RandomQueryInt(_RandomQueryEql):
    def _initialize_query(self, init_v=0.0, v=1.0):
        super(RandomQueryInt, self)._initialize_query(0.0, 1.0)


class RandomQueryFloat(_RandomQueryEql):
    def _initialize_query(self, init_v=0.0, v=1.0):
        super(RandomQueryFloat, self)._initialize_query(0.1, 0.7)


class _RandomQuerySimple(StoreQueryFull):
    def _initialize_query(self, init_v=0.0, v=1.0):
        for k in range(self.search_space.num_solutions):
            query = init_query(self.num_param_per_node, self.steps,init_v=init_v)
            for i in range(self.steps):
                query[i][random.randint(0, self.num_param_per_node[i] - 1)] = v
            self.id_query[k] = query


class RandomQueryIntSimple(_RandomQuerySimple):
    def _initialize_query(self, init_v=0.0, v=1.0):
        super(RandomQueryIntSimple, self)._initialize_query(0.0, 1.0)


class RandomQueryFloatSimple(_RandomQuerySimple):
    def _initialize_query(self, init_v=0.0, v=1.0):
        super(RandomQueryFloatSimple, self)._initialize_query(0.1, 0.7)

# Alias
soft_map_v3 = partial(_soft_map_v3, init_v=0.1)
soft_map_v3_init = _soft_map_v3
map_random_v2 = RandomQueryIntSimple
map_random_v3 = RandomQueryInt
soft_map_random_v2 = RandomQueryFloatSimple
soft_map_random_v3 = RandomQueryFloat
