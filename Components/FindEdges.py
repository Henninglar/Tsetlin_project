def FindEdges(Boardsize):
    edges = [[] for _ in range(Boardsize * Boardsize)]

    def add_edge(node1, node2):
        if node2 not in edges[node1]:
            edges[node1].append(node2)
        if node1 not in edges[node2]:
            edges[node2].append(node1)

    for index in range(Boardsize * Boardsize):
        # Up connection
        if index >= Boardsize:
            add_edge(index, index - Boardsize)

        # Up-right connection
        if index >= Boardsize and index % Boardsize != (Boardsize - 1):
            add_edge(index, index - Boardsize + 1)

        # Down connection
        if index < Boardsize * (Boardsize - 1):
            add_edge(index, index + Boardsize)

        # Down-left connection
        if index < Boardsize * (Boardsize - 1) and index % Boardsize != 0:
            add_edge(index, index + Boardsize - 1)

        # Right connection
        if index % Boardsize != (Boardsize - 1):
            add_edge(index, index + 1)

        # Left connection
        if index % Boardsize != 0:
            add_edge(index, index - 1)

    return edges
