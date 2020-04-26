import numpy as np
import torch as to

BASE_GRAPH = to.tensor([[0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 0, 1],
                        [0, 0, 1, 0]]).float()
BASE_GRAPH_NODE_FEATURES = to.tensor([[1, 2], [1, 1], [2, 0.5], [0.5, 0.5]]).float()
BASE_UNITY_MATRIX = np.ones((BASE_GRAPH_NODE_FEATURES.shape[1], BASE_GRAPH_NODE_FEATURES.shape[1]))
BASE_UNITY_VECTOR = np.ones((BASE_GRAPH_NODE_FEATURES.shape[1]))
BASE_ZEROS_MATRIX = np.zeros((BASE_GRAPH_NODE_FEATURES.shape[1], BASE_GRAPH_NODE_FEATURES.shape[1]))

BASE_GRAPH_EDGE_FEATURES = to.tensor([[[0.0, 0.0],   [1.0, 2.0], [2.0, 0.5],   [0.0, 0.0]],
                                      [[1.0, 2.0],   [0.0, 0.0], [1.0, 1.0],   [0.0, 0.0]],
                                      [[2.0, 0.5],   [1.0, 1.0], [0.0, 0.0],   [0.5, 0.5]],
                                      [[0.0, 0.0],   [0.0, 0.0], [0.5, 0.5],   [0.0, 0.0]]]).float()
BASE_W_MATRIX = to.tensor([[BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX],
                           [BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX],
                           [BASE_UNITY_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX],
                           [BASE_ZEROS_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX]]).float()
BASE_U_MATRIX = to.tensor([BASE_UNITY_MATRIX,
                           BASE_UNITY_MATRIX,
                           BASE_UNITY_MATRIX,
                           BASE_UNITY_MATRIX]).float()

BASE_B_VECTOR = to.tensor(BASE_UNITY_VECTOR).float()

MULTIPLICATION_FACTOR = 0.1
