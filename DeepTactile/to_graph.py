import torch
from torch_geometric.data import Data
import numpy as np
import scipy.spatial as ss


class TactileGraph:
    """
    Constructs a graph for tactile data based on taxel coordinates. Supports manual, KNN, or MST-based graph construction.

    Args:
        k (int): Number of neighbors for KNN. If 0, uses manual graph construction.
        useKNN (bool): Whether to use KNN for graph construction.
        dist_threshold (float): Distance threshold for adding edges in MST-based graph construction.
    """
    def __init__(self, k=0, useKNN=False, dist_threshold=0):
        # Predefined coordinates of 39 tactile sensors.
        tact_coordinates = np.array([
            [-6, 0], [-5.3, -3], [-5.3, 3], [-4.6, -7.8], [-4.6, 7.8],
            [-3.5, 0], [-3.05, -5.2], [-3.05, 5.2], [-3.1, -1.75], [-3.1, 1.75],
            [-1.6, -8.9], [-1.75, -3], [-1.6, 8.9], [-1.75, 3], [-1.5, 0],
            [-0.7, -1.3], [-0.7, 1.3], [0, -6], [0, 6], [0, -3.5],
            [0, 0], [0, 3.5], [0.8, -1.3], [0.8, 1.3], [1.6, -8.9],
            [1.6, 8.9], [1.5, 0], [1.75, 3], [1.75, -3], [3.05, -5.2],
            [3.1, 1.75], [3.05, 5.2], [3.1, -1.75], [3.6, 0], [4.6, -7.8],
            [4.6, 7.8], [5.3, -3], [5.3, 3], [6, 0]
        ])
        assert k >= 0, 'k must be non-negative'
        self.pos = tact_coordinates

        if k == 0:
            # Manual graph construction
            self.edge_origins = np.array([
                1, 1, 1, 2, 3, 6, 2, 2, 7, 9, 3, 3, 8, 10, 4, 4, 7, 11,
                5, 5, 8, 13, 6, 6, 6, 9, 10, 15, 7, 7, 12, 18, 8, 8, 14, 19,
                9, 9, 12, 16, 10, 10, 14, 15, 11, 11, 18, 25, 12, 12, 16, 20,
                13, 13, 19, 26, 14, 14, 17, 22, 15, 15, 15, 16, 17, 21,
                16, 16, 21, 23, 17, 17, 21, 24, 18, 18, 18, 20, 25, 30, 19, 19,
                19, 22, 26, 32, 20, 20, 23, 29, 21, 21, 21, 23, 24, 27, 22,
                28, 23, 23, 27, 29, 24, 24, 24, 27, 28, 31, 25, 35, 26, 36,
                27, 27, 33, 34, 28, 28, 31, 32, 29, 29, 30, 33, 30, 30, 35,
                37, 31, 31, 34, 38, 32, 32, 36, 38, 33, 33, 34, 37, 34, 39,
                37, 39, 38, 39
            ]) - 1
            self.edge_ends = np.array([
                2, 3, 6, 1, 1, 1, 7, 9, 2, 2, 8, 10, 3, 3, 7, 11, 4, 4,
                8, 13, 5, 5, 9, 10, 15, 6, 6, 6, 12, 18, 7, 7, 14, 19, 8, 8,
                12, 16, 9, 9, 14, 15, 10, 10, 18, 25, 11, 11, 16, 20, 12, 12,
                19, 26, 13, 13, 17, 22, 14, 14, 16, 17, 21, 15, 15, 15, 21,
                23, 16, 16, 21, 24, 17, 17, 20, 25, 30, 18, 18, 18, 22, 26,
                32, 19, 19, 19, 23, 29, 20, 20, 23, 24, 27, 21, 21, 21, 28,
                22, 27, 29, 23, 23, 27, 28, 31, 24, 24, 24, 35, 25, 36, 26,
                33, 34, 27, 27, 31, 32, 28, 28, 30, 33, 29, 29, 35, 37, 30,
                30, 34, 38, 31, 31, 36, 38, 32, 32, 34, 37, 33, 33, 39, 34,
                39, 37, 39, 38
            ]) - 1
        elif useKNN:
            # KNN-based graph construction
            tree = ss.KDTree(tact_coordinates)
            _, idxs = tree.query(tact_coordinates, k=k + 1)  # k+1 to include itself
            idxs = idxs[:, 1:]  # Remove self-loops
            edge_origins = np.repeat(np.arange(len(tact_coordinates)), k)
            edge_ends = np.reshape(idxs, (-1))

            # Make the graph undirected
            self.edge_origins = np.hstack((edge_origins, edge_ends))
            self.edge_ends = np.hstack((edge_ends, edge_origins))
        else:
            # MST + threshold-based graph construction
            coordinates = self.pos
            N = len(coordinates)
            self.edge_origins = []
            self.edge_ends = []

            visited_nodes = [20]  # Start from the center node
            unvisited_nodes = list(range(N))
            unvisited_nodes.remove(20)

            # Build MST using Kruskal's algorithm
            while unvisited_nodes:
                min_dist = float('inf')
                origin_index = -1
                end_index = -1
                for node in visited_nodes:
                    dist = torch.norm(
                        torch.tensor(coordinates[unvisited_nodes]) - torch.tensor(coordinates[node]), dim=1
                    )
                    dist_min, idx = torch.sort(dist)
                    if dist_min[0] < min_dist:
                        min_dist = dist_min[0]
                        origin_index = node
                        end_index = unvisited_nodes[idx[0]]
                if end_index >= 0:
                    self.edge_origins.extend([origin_index, end_index])
                    self.edge_ends.extend([end_index, origin_index])
                    visited_nodes.append(end_index)
                    unvisited_nodes.remove(end_index)

            # Add edges within the distance threshold
            for i, j in torch.combinations(torch.arange(N), 2):
                dist = torch.norm(torch.tensor(coordinates[i]) - torch.tensor(coordinates[j]))
                if dist < dist_threshold:
                    self.edge_origins.extend([i, j])
                    self.edge_ends.extend([j, i])

    def get_edge(self):
        """
        Returns the edge indices of the graph.

        Returns:
            torch.Tensor: Edge indices as [2, num_edges].
        """
        return torch.tensor([self.edge_origins, self.edge_ends])

    def __call__(self, sample):
        """
        Constructs a graph object using the input sample as node features.

        Args:
            sample (torch.Tensor): Node feature tensor.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        graph_x = sample
        graph_edge_index = torch.tensor([self.edge_origins, self.edge_ends], dtype=torch.long)
        graph_pos = self.pos
        data = Data(x=graph_x, edge_index=graph_edge_index, pos=graph_pos)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}"
