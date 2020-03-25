from typing import List

from src.data.test_training_dataset import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES
from src.domain.graph import Graph
from src.repository.interface.repository import Repository


class TrainingDataRepository(Repository):
    def __init__(self) -> None:
        super().__init__()

    def get_all(self) -> List[Graph]:
        return [Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)]
