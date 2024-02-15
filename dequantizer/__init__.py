import os

os.environ["JAX_ENABLE_X64"] = "True"

from .src.graph import (
    TensorGraph,
    get_nd_lattice,
    get_random_tree_tensor_graph,
)

from .src.mappings import get_belief_propagation_map
from .src.mappings import get_vidal_gauge_fixing_map
from .src.mappings import get_vidal_gauge_distance_map
from .src.mappings import get_symmetric_gauge_fixing_map
from .src.mappings import messages_frob_distance
