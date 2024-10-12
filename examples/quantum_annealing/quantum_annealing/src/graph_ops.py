from typing import List, Tuple, Set
from collections import deque


def _pairs2graph(pairs: List[Tuple[int, int]]) -> List[List[int]]:
    size = max(map(lambda x: max(x), pairs)) + 1
    graph: List[List[int]] = [[] for _ in range(size)]
    for lhs, rhs in pairs:
        graph[lhs].append(rhs)
        graph[rhs].append(lhs)
    return graph


def _shortest_cycle_length_trough_edge(
    graph: List[List[int]],
    src: int,
    dst: int,
) -> int:
    if dst in graph[src]:
        nodes_queue = deque(((src, 0),))
        visited_labels = set()
        while nodes_queue:
            curr_node = nodes_queue.popleft()
            curr_label = curr_node[0]
            curr_path_length = curr_node[1]
            if curr_label != dst:
                if curr_label not in visited_labels:
                    visited_labels.add(curr_label)
                    for neighboring_label in filter(
                        lambda x: (curr_label, x) != (src, dst), graph[curr_label]
                    ):
                        nodes_queue.append((neighboring_label, curr_path_length + 1))
            else:
                return curr_path_length + 1
        return -1
    else:
        raise ValueError(f"There is no edge from {src} to {dst}")


def shortest_cycles_distr(pairs: List[Tuple[int, int]]) -> List[float]:
    edges_number = len(pairs)
    cycle_lengths_stat: List[float] = []
    visited_edges: Set[bool] = set()
    graph = _pairs2graph(pairs)
    for curr_label, neighboring_label in pairs:
        if (curr_label, neighboring_label) not in visited_edges and (
            neighboring_label,
            curr_label,
        ) not in visited_edges:
            length = _shortest_cycle_length_trough_edge(
                graph, curr_label, neighboring_label
            )
            if len(cycle_lengths_stat) > length:
                cycle_lengths_stat[length] += 1.0 / edges_number
            else:
                curr_length = len(cycle_lengths_stat)
                cycle_lengths_stat.extend(
                    [0.0] * (length - curr_length) + [1.0 / edges_number]
                )
    return cycle_lengths_stat[1:]
