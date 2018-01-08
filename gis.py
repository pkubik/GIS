from enum import Enum
from queue import PriorityQueue


class PathMeta:
    def __init__(self, pre: int, post: int):
        """
        Metadata of the path passing through given edge

        `pre` + `post` + 1 == length of the whole path

        :param pre: length of the part of the path that occur before given edge
        :param post: length of the part of the path that occur after given edge
        """
        self.pre = pre
        self.post = post

    def __repr__(self):
        return 'meta' + str((self.pre, self.post))


class Vertex:
    def __init__(self, id_):
        self.id = id_
        self.edges = set()

    def __repr__(self):
        return 'V' + str(self.id)


class Edge:
    def __init__(self, source: Vertex, destination: Vertex, path_meta: PathMeta = None):
        self.source = source
        self.destination = destination
        self.path_meta = path_meta

    def available_flow(self) -> int:
        if self.path_meta is None:
            return 1
        else:
            return 0

    def is_valid_continuation_of(self, edge: 'Edge'):
        return self.path_meta.pre == edge.path_meta.pre + 1

    def __repr__(self):
        return '{}-{}: {}'.format(self.source, self.destination, self.path_meta)


class Graph:
    def __init__(self):
        self.vertices = {}
        self.source_id = 0
        self.sink_id = 0
        self.edges = []  # just for debugging

    def add_edge(self, source_id: int, destination_id: int):
        if source_id not in self.vertices:
            self.vertices[source_id] = Vertex(source_id)
        source = self.vertices[source_id]

        if destination_id not in self.vertices:
            self.vertices[destination_id] = Vertex(destination_id)
        destination = self.vertices[destination_id]

        edge = Edge(source, destination)
        source.edges.add(edge)
        destination.edges.add(edge)
        self.sink_id = max(destination_id, self.sink_id)
        self.edges.append(edge)

    def add_path_meta(self, source_id: int, destination_id: int, path_meta: PathMeta):
        source = self.vertices[source_id]
        valid_edges = {e for e in source.edges if e.destination.id == destination_id}
        assert len(valid_edges) == 1  # assume only one edge in given direction
        edge = next(iter(valid_edges))
        edge.path_meta = path_meta


class StepType(Enum):
    NORMAL = 0
    BACKWARD = 1
    REPLACE = 2


class Node:
    def __init__(self,
                 vertex: Vertex,
                 path_length: int,
                 max_path_length: int,
                 step_type: StepType,
                 parent: 'Node' = None,
                 edge: 'Edge' = None,
                 allow_residual=False):
        self.vertex = vertex
        self.path_length = path_length
        self.max_path_length = max_path_length
        self.step_type = step_type
        self.parent = parent
        self.edge = edge
        self.allow_residual = allow_residual

    def simple_next_node(self, edge: Edge):
        """
        Simply create the next node like in BFS

        :param edge: Edge used to extend the current path
        :return: Node
        """
        node = Node.__new__(Node)
        node.vertex = edge.destination
        node.path_length = self.path_length + 1
        node.max_path_length = self.max_path_length
        node.step_type = StepType.NORMAL
        node.parent = self
        node.edge = edge
        node.allow_residual = False
        return node

    def replacing_node(self, edge: Edge):
        """
        :param edge: Initial edge of the subpath to be used
        :return: Node
        """
        node = Node.__new__(Node)
        node.vertex = self.vertex
        node.path_length = edge.path_meta.pre
        node.max_path_length = max(self.max_path_length, self.path_length + 1 + edge.path_meta.post)
        node.step_type = StepType.REPLACE
        node.parent = self
        node.edge = edge
        node.allow_residual = True
        return node

    def backward_node(self, edge: Edge):
        """
        Node created by going backward through residual connection

        :param edge: residual edge to follow
        :return: Node
        """

        node = Node.__new__(Node)
        node.vertex = edge.source
        node.path_length = self.path_length - 1
        node.max_path_length = self.max_path_length
        node.step_type = StepType.BACKWARD
        node.parent = self
        node.edge = edge
        node.allow_residual = True
        return node

    def __lt__(self, other):
        return max(self.max_path_length, self.path_length) < max(other.max_path_length, other.path_length)

    def __repr__(self):
        return 'node({}, {}, {}, {})'.format(self.vertex, self.path_length, self.max_path_length, self.allow_residual)


def next_valid_edge_to_update(edge: Edge):
    candidates = [e for e in edge.destination.edges
                  if e.source is edge.destination  # only starting at given edge destination
                  and e.path_meta.pre == edge.path_meta.pre + 1]  # extends by one
    if len(candidates) == 0:
        return None
    else:
        return candidates[0]


def apply_path(node: Node):
    current = node
    post = 0
    while current.parent is not None:
        print(current, current.edge)

        if current.step_type == StepType.NORMAL:
            # we annotate the edge with the new path information
            assert post >= 0  # this branch can not execute after residual update
            current.edge.path_meta = PathMeta(current.path_length - 1, post)
        elif current.step_type == StepType.BACKWARD:
            # we remove path from the residual connection we used (it's no longer residual)
            current.edge.path_meta = None
            post = -1  # just for assertion
        else:
            # we have to recursively go to the sink and update the edges
            path_length = current.parent.path_length
            edge = current.edge
            post = edge.path_meta.post
            while edge is not None:
                print(edge, end=' | ')
                next_edge = next_valid_edge_to_update(edge)
                edge.path_meta.pre = path_length
                path_length += 1
                edge = next_edge
            print()

        current = current.parent
        post += 1


def iterate(graph: Graph):
    source = graph.vertices[graph.source_id]
    sink = graph.vertices[graph.sink_id]
    initial_node = Node(source, 0, 0, StepType.NORMAL)

    q = PriorityQueue()
    q.put(initial_node)

    best_by_vertex = {}  # helper dict to avoid pushing redundant nodes

    def push(node: Node):
        key = (node.vertex.id, node.allow_residual)
        if key not in best_by_vertex:
            best_by_vertex[key] = 2**31
        current_best = best_by_vertex[key]

        node_score = max(node.path_length, node.max_path_length)
        if current_best >= node_score:
            q.put(node)
            best_by_vertex[key] = node_score

    current = None
    while not q.empty():
        current = q.get()
        assert isinstance(current, Node)
        print(current, ':', q.queue)

        if current.vertex is sink:
            print("FOUND THE PATH")
            break

        for e in current.vertex.edges:
            assert isinstance(e, Edge)
            if e.source is current.vertex:  # edge is starting at the current vertex
                if e.available_flow() > 0:  # the edge is free
                    node = current.simple_next_node(e)
                    push(node)
                else:  # the edge is already used by another path
                    not_source = current.vertex is not source  # pointless to go all way back to the source
                    if not_source:  # only source may have `parent == None`
                        not_to_parent = current.parent.vertex is not e.destination
                        not_twice = current.parent.vertex is not current.vertex
                    else:
                        not_to_parent = False
                        not_twice = False
                    if not_to_parent and not_twice:
                        node = current.replacing_node(e)
                        push(node)
            else:  # edge is ending at the current vertex
                if e.path_meta is not None:
                    same_path = e.path_meta.pre + 1 == current.path_length  # same path which was switched earlier
                else:
                    same_path = False
                not_to_source = e.source is not source  # don't go back to the source
                if current.allow_residual and same_path and not_to_source:
                    push(current.backward_node(e))
        current = None

    if current is not None:
        apply_path(current)
        return True
    else:
        return False


def get_paths(graph: Graph):
    source = graph.vertices[graph.source_id]
    paths = []
    used_edges = set()

    for se in source.edges:
        if se.source == source:
            path = [se]
            while path[-1].destination.id != graph.sink_id:
                last_edge = path[-1]
                last_vertex = path[-1].destination
                for e in last_vertex.edges:
                    if (e.source == last_vertex
                            and e.is_valid_continuation_of(last_edge)
                            and e not in used_edges):
                        path.append(e)
                        used_edges.add(e)
                        break
            paths.append(path)

    return paths


def edge_seq_to_vertex_seq(seq: list):
    return [seq[0].source] + [e.destination for e in seq]


def main():
    target_flow = 3
    g = Graph()

    # this should be read from file
    # note that node 7 was replaced with 10 and additional connection 6-8-9-10 has been added to the original example
    """
    # First graph
    edges = [
        (0, 1),
        (0, 3),
        (0, 5),
        (1, 2),
        (1, 4),
        (2, 10),
        (3, 4),
        (3, 6),
        (4, 10),
        (5, 6),
        (6, 10),

        (6, 8),  # additional connection
        (8, 9),
        (9, 10),

        (0, 7),  # bonus connection to check redundant nodes elimination
        (7, 6)
    ]
    """

    # Second graph
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 2),
        (2, 4)
    ]

    # build graph from input
    for e in edges:
        g.add_edge(*e)

    # add some paths manually to recreate the scenario from the docs
    """
    # For first graph
    g.add_path_meta(0, 1, PathMeta(0, 2))
    g.add_path_meta(1, 4, PathMeta(1, 1))
    g.add_path_meta(4, 10, PathMeta(2, 0))

    g.add_path_meta(0, 3, PathMeta(0, 2))
    g.add_path_meta(3, 6, PathMeta(1, 1))
    g.add_path_meta(6, 10, PathMeta(2, 0))
    """

    found_path = True
    iteration_number = 0
    while found_path:
        print("ITERATION", iteration_number)
        found_path = iterate(g)
        print()
        print("Edges after iteration:")
        print(g.edges)
        print()
        print()
        iteration_number += 1

    paths = get_paths(g)
    print("Found paths:")
    for p in paths:
        print(edge_seq_to_vertex_seq(p))


if __name__ == "__main__":
    main()
