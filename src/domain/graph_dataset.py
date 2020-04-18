from torch.utils.data import Dataset

from src.domain.graph import Graph


class GraphDataset(Dataset):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem(self, idx: int) -> Graph:
        pass
