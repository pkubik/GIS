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


class Node:
    def __init__(self,
                 vertex: Vertex,
                 path_length: int,
                 max_path_length: int,
                 parent: 'Node' = None,
                 edge_to_update: 'Edge' = None,
                 allow_residual=False):
        self.vertex = vertex
        self.path_length = path_length
        self.max_path_length = max_path_length
        self.parent = parent
        self.edge_to_update = edge_to_update
        self.allow_residual = allow_residual

    def simple_next_node(self, destination: Vertex):
        """
        Simply create the next node like in BFS

        :param destination: vertex to be added to the path
        :return: Node
        """
        node = Node.__new__(Node)
        node.vertex = destination
        node.path_length = self.path_length + 1
        node.max_path_length = self.max_path_length
        node.parent = self
        node.edge_to_update = None
        node.allow_residual = False
        return node

    def replacing_node(self, edge: Edge):
        """
        :param edge: Initial edge of the subpath to be used
        :return: Node
        """
        node = Node.__new__(Node)
        node.vertex = self.vertex
        node.path_length = self.path_length
        node.max_path_length = max(self.max_path_length, self.path_length + 1 + edge.path_meta.post)
        node.parent = self
        node.edge_to_update = edge
        node.allow_residual = True
        return node

    def backward_node(self, vertex: Vertex):
        """
        Node created by going backward through residual connection

        :param vertex: source vertex of the residual edge
        :return: Node
        """

        node = Node.__new__(Node)
        node.vertex = vertex
        node.path_length = self.path_length - 1
        node.max_path_length = self.max_path_length
        node.parent = self
        node.edge_to_update = None
        node.allow_residual = True
        return node

    def from_normal_edge(self, edge: Edge):
        if edge.available_flow() > 0:
            return self.simple_next_node(edge.destination)
        else:
            return self.replacing_node(edge)

    def from_residual_edge(self, edge: Edge):
        return self.backward_node(edge.source)

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
        print(current, current.edge_to_update)

        if current.edge_to_update is None:
            used_residual_connection = current.path_length == current.parent.path_length - 1
            if used_residual_connection:
                # we remove path from the residual connection we used (it's no longer residual)
                edge = [e for e in current.vertex.edges if e.destination is current.parent.vertex][0]
                edge.path_meta = None
                post = -1  # just for assertion
            else:
                assert post >= 0  # this branch can not execute after residual update
                # we annotate the edge with the new path information
                edge = [e for e in current.vertex.edges if e.source is current.parent.vertex][0]
                edge.path_meta = PathMeta(current.path_length - 1, post)
        else:
            path_length = current.parent.path_length
            edge = current.edge_to_update
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
    initial_node = Node(source, 0, 0)

    q = PriorityQueue()
    q.put(initial_node)

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
                    node = current.simple_next_node(e.destination)
                    q.put(node)
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
                        q.put(node)
            else:  # edge is ending at the current vertex
                if e.path_meta is not None:
                    same_path = e.path_meta.pre + 1 == current.path_length  # same path which was switched earlier
                else:
                    same_path = False
                not_to_source = e.source is not source  # don't go back to the source
                if current.allow_residual and same_path and not_to_source:
                    q.put(current.backward_node(e.source))

    apply_path(current)


def main():
    target_flow = 3
    g = Graph()

    # this should be read from file
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

        (6, 8),
        (8, 9),
        (9, 10)
    ]

    # build graph from input
    for e in edges:
        g.add_edge(*e)

    # add some paths manually to recreate the scenario from the docs
    g.add_path_meta(0, 1, PathMeta(0, 2))
    g.add_path_meta(1, 4, PathMeta(1, 1))
    g.add_path_meta(4, 10, PathMeta(2, 0))

    g.add_path_meta(0, 3, PathMeta(0, 2))
    g.add_path_meta(3, 6, PathMeta(1, 1))
    g.add_path_meta(6, 10, PathMeta(2, 0))

    iterate(g)

    print()

    print(g.edges)


if __name__ == "__main__":
    main()
