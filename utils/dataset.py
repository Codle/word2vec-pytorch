from typing import List, Union
from torch.utils.data import Dataset
from vocab import Vocab


class CorpusDataset(Dataset):

    def __init__(self, data: List[List[int]], win_size=4) -> None:
        """Init dataset, note we used a list[list] type data because we still
        want to have the information of pargraph.

        Arguments:
            data {List[List[int]]} -- [description]

        Keyword Arguments:
            win_size {int} -- [description] (default: {4})
        """
        self.win_size = win_size


    def __getitem__(self, item):
        # padding
        

    def __len__(self):
        pass
