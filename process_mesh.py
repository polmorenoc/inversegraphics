import meshtool.filters.simplify_filters.sander_simplify
from meshtool.filters.base_filters import MetaFilter
from meshtool.filters import factory
from itertools import chain, izip, combinations
import collada
import numpy


def process(meshPath):

    mesh = collada.Collada(meshPath)

    # 'triangulate',
    optimize_filters = [
                        'combine_primitives',
                        'optimize_sources',
                        'strip_unused_sources',
                        'normalize_indices'
                        ]

    for f in optimize_filters:
        inst = factory.getInstance(f)
        mesh = inst.apply(mesh)

    # f = 'sander_simplify'
    # pmout = open('pm_file', 'w')
    # inst = factory.getInstance(f)
    # mesh = inst.apply(mesh, pmout)

    # s = meshtool.filters.simplify_filters.sander_simplify.SanderSimplify(mesh, pmout)

    # meshsimple = s.simplify()

    return mesh


