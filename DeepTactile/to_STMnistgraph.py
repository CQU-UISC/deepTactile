import torch
from torch_geometric.data import Data
import numpy as np
import scipy.spatial as ss

class TactileGraphSTMnist:
    """
    Constructs a graph for tactile ST-MNIST data based on taxel coordinates.
    Supports manual, KNN, or MST-based graph construction.

    Args:
        k (int): Number of neighbors for KNN. If 0, uses manual graph construction.
        useKNN (bool): Whether to use KNN for graph construction.
        dist_threshold (float): Distance threshold for adding edges in MST-based graph construction.
    """
    def __init__(self, k=0, useKNN=False, dist_threshold=0):
        # Predefined coordinates for ST-MNIST tactile sensors (10x10 grid).
        tact_coordinates = np.array([
            [x, y] for y in range(1, 11) for x in range(1, 11)
        ])
        assert k >= 0, 'k must be non-negative'
        self.pos = tact_coordinates

        if k == 0:
            # Manual graph construction
            self.edge_origins = np.array([1, 1, 1,
                                          2, 2, 2, 2, 2,
                                          3, 3, 3, 3, 3,
                                          4, 4, 4, 4, 4,
                                          5, 5, 5, 5, 5,
                                          6, 6, 6, 6, 6,
                                          7, 7, 7, 7, 7,
                                          8, 8, 8, 8, 8,
                                          9, 9, 9, 9, 9,
                                          10, 10, 10,
                                          11, 11, 11, 11, 11,
                                          12, 12, 12, 12, 12, 12, 12, 12,
                                          13, 13, 13, 13, 13, 13, 13, 13,
                                          14, 14, 14, 14, 14, 14, 14, 14,
                                          15, 15, 15, 15, 15, 15, 15, 15,
                                          16, 16, 16, 16, 16, 16, 16, 16,
                                          17, 17, 17, 17, 17, 17, 17, 17,
                                          18, 18, 18, 18, 18, 18, 18, 18,
                                          19, 19, 19, 19, 19, 19, 19, 19,
                                          20, 20, 20, 20, 20,
                                          21, 21, 21, 21, 21,
                                          22, 22, 22, 22, 22, 22, 22, 22,
                                          23, 23, 23, 23, 23, 23, 23, 23,
                                          24, 24, 24, 24, 24, 24, 24, 24,
                                          25, 25, 25, 25, 25, 25, 25, 25,
                                          26, 26, 26, 26, 26, 26, 26, 26,
                                          27, 27, 27, 27, 27, 27, 27, 27,
                                          28, 28, 28, 28, 28, 28, 28, 28,
                                          29, 29, 29, 29, 29, 29, 29, 29,
                                          30, 30, 30, 30, 30,
                                          31, 31, 31, 31, 31,
                                          32, 32, 32, 32, 32, 32, 32, 32,
                                          33, 33, 33, 33, 33, 33, 33, 33,
                                          34, 34, 34, 34, 34, 34, 34, 34,
                                          35, 35, 35, 35, 35, 35, 35, 35,
                                          36, 36, 36, 36, 36, 36, 36, 36,
                                          37, 37, 37, 37, 37, 37, 37, 37,
                                          38, 38, 38, 38, 38, 38, 38, 38,
                                          39, 39, 39, 39, 39, 39, 39, 39,
                                          40, 40, 40, 40, 40,
                                          41, 41, 41, 41, 41,
                                          42, 42, 42, 42, 42, 42, 42, 42,
                                          43, 43, 43, 43, 43, 43, 43, 43,
                                          44, 44, 44, 44, 44, 44, 44, 44,
                                          45, 45, 45, 45, 45, 45, 45, 45,
                                          46, 46, 46, 46, 46, 46, 46, 46,
                                          47, 47, 47, 47, 47, 47, 47, 47,
                                          48, 48, 48, 48, 48, 48, 48, 48,
                                          49, 49, 49, 49, 49, 49, 49, 49,
                                          50, 50, 50, 50, 50,
                                          51, 51, 51, 51, 51,
                                          52, 52, 52, 52, 52, 52, 52, 52,
                                          53, 53, 53, 53, 53, 53, 53, 53,
                                          54, 54, 54, 54, 54, 54, 54, 54,
                                          55, 55, 55, 55, 55, 55, 55, 55,
                                          56, 56, 56, 56, 56, 56, 56, 56,
                                          57, 57, 57, 57, 57, 57, 57, 57,
                                          58, 58, 58, 58, 58, 58, 58, 58,
                                          59, 59, 59, 59, 59, 59, 59, 59,
                                          60, 60, 60, 60, 60,
                                          61, 61, 61, 61, 61,
                                          62, 62, 62, 62, 62, 62, 62, 62,
                                          63, 63, 63, 63, 63, 63, 63, 63,
                                          64, 64, 64, 64, 64, 64, 64, 64,
                                          65, 65, 65, 65, 65, 65, 65, 65,
                                          66, 66, 66, 66, 66, 66, 66, 66,
                                          67, 67, 67, 67, 67, 67, 67, 67,
                                          68, 68, 68, 68, 68, 68, 68, 68,
                                          69, 69, 69, 69, 69, 69, 69, 69,
                                          70, 70, 70, 70, 70,
                                          71, 71, 71, 71, 71,
                                          72, 72, 72, 72, 72, 72, 72, 72,
                                          73, 73, 73, 73, 73, 73, 73, 73,
                                          74, 74, 74, 74, 74, 74, 74, 74,
                                          75, 75, 75, 75, 75, 75, 75, 75,
                                          76, 76, 76, 76, 76, 76, 76, 76,
                                          77, 77, 77, 77, 77, 77, 77, 77,
                                          78, 78, 78, 78, 78, 78, 78, 78,
                                          79, 79, 79, 79, 79, 79, 79, 79,
                                          80, 80, 80, 80, 80,
                                          81, 81, 81, 81, 81,
                                          82, 82, 82, 82, 82, 82, 82, 82,
                                          83, 83, 83, 83, 83, 83, 83, 83,
                                          84, 84, 84, 84, 84, 84, 84, 84,
                                          85, 85, 85, 85, 85, 85, 85, 85,
                                          86, 86, 86, 86, 86, 86, 86, 86,
                                          87, 87, 87, 87, 87, 87, 87, 87,
                                          88, 88, 88, 88, 88, 88, 88, 88,
                                          89, 89, 89, 89,  89, 89, 89, 89,
                                          90, 90, 90, 90, 90,
                                          91, 91, 91,
                                          92, 92, 92, 92, 92,
                                          93, 93, 93, 93, 93,
                                          94, 94, 94, 94, 94,
                                          95, 95, 95, 95, 95,
                                          96, 96, 96, 96, 96,
                                          97, 97, 97, 97, 97,
                                          98, 98, 98, 98, 98,
                                          99, 99, 99, 99, 99,
                                          100, 100, 100]) - 1  # Adjust indices to 0-based
            self.edge_ends = np.array([2, 11, 12,
                                       1, 11, 12, 13, 3,
                                       2, 12, 13, 14, 4,
                                       3, 13, 14, 15, 5,
                                       4, 14, 15, 16, 6,
                                       5, 15, 16, 17, 7,
                                       6, 16, 17, 18, 8,
                                       7, 17, 18, 19, 9,
                                       8, 18, 19, 20, 10,
                                       9, 19, 20,
                                       1, 2, 12, 22, 21,
                                       1, 2, 3, 11, 13, 21, 22, 23,
                                       2, 3, 4, 12, 14, 22, 23, 24,
                                       3, 4, 5, 13, 15, 23, 24, 25,
                                       4, 5, 6, 14, 16, 24, 25, 26,
                                       5, 6, 7, 15, 17, 25, 26, 27,
                                       6, 7, 8, 16, 18, 26, 27, 28,
                                       7, 8, 9, 17, 19, 27, 28, 29,
                                       8, 9, 10, 18, 20, 28, 29, 30,
                                       9, 10, 19, 29, 30,
                                       11, 12, 22, 31, 32,
                                       11, 12, 13, 21, 23, 31, 32, 33,
                                       12, 13, 14, 22, 24, 32, 33, 34,
                                       13, 14, 15, 23, 25, 33, 34, 35,
                                       14, 15, 16, 24, 26, 34, 35, 36,
                                       15, 16, 17, 25, 27, 35, 36, 37,
                                       16, 17, 18, 26, 28, 36, 37, 38,
                                       17, 18, 19, 27, 29, 37, 38, 39,
                                       18, 19, 20, 28, 30, 38, 39, 40,
                                       19, 20, 29, 39, 40,
                                       21, 22, 32, 41, 42,
                                       21, 22, 23, 31, 33, 41, 42, 43,
                                       22, 23, 24, 32, 34, 42, 43, 44,
                                       23, 24, 25, 33, 35, 43, 44, 45,
                                       24, 25, 26, 34, 36, 44, 45, 46,
                                       25, 26, 27, 35, 37, 45, 46, 47,
                                       26, 27, 28, 36, 38, 46, 47, 48,
                                       27, 28, 29, 37, 39, 47, 48, 49,
                                       28, 29, 30, 38, 40, 48, 49, 50,
                                       29, 30, 39, 49, 50,
                                       31, 32, 42, 51, 52,
                                       31, 32, 33, 41, 43, 51, 52, 53,
                                       32, 33, 34, 42, 44, 52, 53, 54,
                                       33, 34, 35, 43, 45, 53, 54, 55,
                                       34, 35, 36, 44, 46, 54, 55, 56,
                                       35, 36, 37, 45, 47, 55, 56, 57,
                                       36, 37, 38, 46, 48, 56, 57, 58,
                                       37, 38, 39, 47, 49, 57, 58, 59,
                                       38, 39, 40, 48, 50, 58, 59, 60,
                                       39, 40, 49, 59, 60,
                                       41, 42, 52, 61, 62,
                                       41, 42, 43, 51, 53, 61, 62, 63,
                                       42, 43, 44, 52, 54, 62, 63, 64,
                                       43, 44, 45, 53, 55, 63, 64, 65,
                                       44, 45, 46, 54, 56, 64, 65, 66,
                                       45, 46, 47, 55, 57, 65, 66, 67,
                                       46, 47, 48, 56, 58, 66, 67, 68,
                                       47, 48, 49, 57, 59, 67, 68, 69,
                                       48, 49, 50, 58, 60, 58, 59, 70,
                                       49, 50, 59, 69, 70,
                                       51, 52, 62, 71, 72,
                                       51, 52, 53, 61, 63, 71, 72, 73,
                                       52, 53, 54, 62, 64, 72, 73, 74,
                                       53, 54, 55, 63, 65, 73, 74, 75,
                                       54, 55, 56, 64, 66, 74, 75, 76,
                                       55, 56, 57, 65, 67, 75, 76, 77,
                                       56, 57, 58, 66, 68, 76, 77, 78,
                                       57, 58, 59, 67, 69, 77, 78, 79,
                                       58, 59, 60, 68, 70, 78, 79, 80,
                                       59, 60, 69, 79, 80,
                                       61, 62, 72, 81, 82,
                                       61, 62, 63, 71, 73, 81, 82, 83,
                                       62, 63, 64, 72, 74, 82, 83, 83,
                                       63, 64, 65, 73, 75, 83, 84, 85,
                                       64, 65, 66, 74, 76, 84, 85, 86,
                                       65, 66, 67, 75, 77, 85, 86, 87,
                                       66, 67, 68, 76, 78, 86, 87, 88,
                                       67, 68, 69, 77, 79, 87, 88, 89,
                                       68, 69, 70, 78, 80, 88, 89, 90,
                                       69, 70, 79, 89, 90,
                                       71, 72, 82, 91, 92,
                                       71, 72, 73, 81, 83, 91, 92, 93,
                                       72, 73, 74, 82, 84, 92, 93, 94,
                                       73, 74, 75, 83, 85, 93, 94, 95,
                                       74, 75, 76, 84, 86, 94, 95, 96,
                                       75, 76, 77, 85, 87, 95, 96, 97,
                                       76, 77, 78, 86, 88, 96, 97, 98,
                                       77, 78, 79, 87, 89, 97, 98, 99,
                                       78, 79, 80, 88, 90, 98, 99, 100,
                                       79, 80, 89, 99, 100,
                                       81, 82, 92,
                                       81, 82, 83, 93, 91,
                                       82, 83, 84, 92, 94,
                                       83, 84, 85, 93, 95,
                                       84, 85, 86, 94, 96,
                                       85, 86, 87, 95, 97,
                                       86, 87, 88, 96, 98,
                                       87, 88, 89, 97, 99,
                                       88, 89, 90, 98, 100,
                                       89, 90, 99]) - 1
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

            visited_nodes = [55]  # Start from the center node
            unvisited_nodes = list(range(N))
            unvisited_nodes.remove(55)

            # Build MST using Kruskal's algorithm
            while unvisited_nodes:
                min_dist = float('inf')
                origin_index = -1
                end_index = -1
                for node in visited_nodes:
                    dist = torch.norm(
                        torch.tensor(coordinates[unvisited_nodes]) - torch.tensor(coordinates[node]),
                        dim=1
                    )
                    dist_min, idx = torch.sort(dist)
                    if dist_min[0] < min_dist:
                        min_dist = dist_min[0]
                        origin_index = node
                        end_index = unvisited_nodes[idx[0]]
                if end_index >= 0:
                    # Add edge (undirected)
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


if __name__ == '__main__':
    graph = TactileGraphSTMnist(k=3)
    print(graph)
