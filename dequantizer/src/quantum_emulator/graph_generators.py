from ..graph import TensorGraph

"""This function generates a cube with additional
qubits on edges. It should be simple for belief propagation methods
to converge to almost numerically exact solution.
It serves only for testing needs."""


def _heavy_cube_graph() -> TensorGraph:
    processor = TensorGraph()
    for _ in range(20):
        processor.add_node()
    processor.add_edge((0, 1), 1)
    processor.add_edge((1, 2), 1)
    processor.add_edge((2, 3), 1)
    processor.add_edge((3, 4), 1)
    processor.add_edge((4, 5), 1)
    processor.add_edge((5, 6), 1)
    processor.add_edge((6, 7), 1)
    processor.add_edge((7, 0), 1)
    processor.add_edge((12, 13), 1)
    processor.add_edge((13, 14), 1)
    processor.add_edge((14, 15), 1)
    processor.add_edge((15, 16), 1)
    processor.add_edge((16, 17), 1)
    processor.add_edge((17, 18), 1)
    processor.add_edge((18, 19), 1)
    processor.add_edge((19, 12), 1)
    processor.add_edge((0, 8), 1)
    processor.add_edge((2, 9), 1)
    processor.add_edge((4, 10), 1)
    processor.add_edge((6, 11), 1)
    processor.add_edge((8, 12), 1)
    processor.add_edge((9, 14), 1)
    processor.add_edge((10, 16), 1)
    processor.add_edge((11, 18), 1)
    return processor


"""Two-cells heavy hex lattice. It serves for testing purposes."""


def _small_heavy_hex() -> TensorGraph:
    lattice = TensorGraph()
    for _ in range(21):
        lattice.add_node()
    for i in range(4):
        lattice.add_edge((i, i + 1), 1)
    for i in range(5, 11):
        lattice.add_edge((i, i + 1), 1)
    for i in range(12, 16):
        lattice.add_edge((i, i + 1), 1)
    lattice.add_edge((0, 17), 1)
    lattice.add_edge((17, 7), 1)
    lattice.add_edge((4, 18), 1)
    lattice.add_edge((18, 11), 1)
    lattice.add_edge((5, 19), 1)
    lattice.add_edge((19, 12), 1)
    lattice.add_edge((10, 20), 1)
    lattice.add_edge((20, 16), 1)
    return lattice
