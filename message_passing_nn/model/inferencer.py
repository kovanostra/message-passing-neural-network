from typing import List, Tuple

import torch as to
from torch import nn
from torch.utils.data import DataLoader

from message_passing_nn.data import DataPreprocessor


class Inferencer:
    def __init__(self, data_preprocessor: DataPreprocessor, device: str, normalize: bool = False) -> None:
        self.preprocessor = data_preprocessor
        self.device = device
        self.normalize = normalize

    def do_inference(self, model: nn.Module, inference_data: DataLoader) -> List[Tuple[to.Tensor, to.Tensor]]:
        outputs_labels_pairs = []
        with to.no_grad():
            for features, labels in inference_data:
                node_features, adjacency_matrix = features
                node_features, adjacency_matrix, labels = node_features.to(self.device), \
                                                          adjacency_matrix.to(self.device), \
                                                          labels.to(self.device)
                if self.normalize:
                    node_features = self.preprocessor.normalize(node_features, self.device)
                    labels = self.preprocessor.normalize(labels, self.device)
                outputs = model.forward(node_features, adjacency_matrix, batch_size=1)
                outputs_labels_pairs.append((outputs, labels))
        return outputs_labels_pairs
