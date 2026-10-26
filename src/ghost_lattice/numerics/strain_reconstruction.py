"""The existing bond least-squares strain, with arbitrary bond storage order.

compact_rotated.strain_operator requires contiguous owner groups. Poly2D.j04
stores query-pair directions without owner sorting. Reindex only the output
operator's columns; the physical model operators are never reordered.
"""

from types import SimpleNamespace
import numpy as np
from .rotated_plane import strain_operator


def strain_operator_any_order(m):
    order = np.argsort(m.I, kind="stable")
    proxy = SimpleNamespace(
        N=m.N, nb=m.nb, I=m.I[order], L0=m.L0[order], nnL=m.nnL[order]
    )
    G = strain_operator(proxy)
    G.indices = order[G.indices].astype(G.indices.dtype)
    G.has_sorted_indices = False
    G.sort_indices()
    return G
