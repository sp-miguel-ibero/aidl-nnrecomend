import torch.utils.data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np
from typing import Container
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    """
    basic dataset class
    """
    def __init__(self, interactions: np.ndarray):
        """
        :param interactions: array with 3 columns (user id, item id, label)
        """
        self.__interactions, self.idsize = self.__preprocess(interactions)

    @staticmethod
    def __preprocess(interactions: np.ndarray):
        """
        preprocesses interactions so that user ids start with 0 and item ids start after the highest user id
        """
        interactions = np.array(interactions).astype(int)
        assert len(interactions.shape) == 2 # should be two dimensions
        if interactions.shape[1] == 2:
            # if the interactions don't come with label column, create it with ones
            interactions = np.c_[interactions, np.ones(interactions.shape[0], int)]
        assert interactions.shape[1] > 2 # should have at least 3 columns
        interactions[:, :2] -= np.min(interactions[:, :2], axis=0) # guarantee that ids start with 0
        idsize = np.max(interactions[:, :2], axis=0) + 1 
        # guarantee that item ids start after the last user id
        idsize[1] += idsize[0]
        interactions[:, 1] += idsize[0]
        return interactions, idsize

    def __len__(self):
        return len(self.__interactions)

    def __getitem__(self, index):
        return self.__interactions[index]

    def __get_random_item(self) -> int:
        """
        return a valid random item id
        """
        return np.random.randint(self.idsize[0], self.idsize[1])

    def get_random_negative_item(self, user: int, item: int, container: Container) -> int:
        """
        return a random item id that meets certain conditions
        TODO: this method can produce infinite loops if there is no item that meets the requirements

        :param user: should not have an interaction with the given user
        :param item: should not be this item
        :param container: container to check if the interaction exists (usually the adjacency matrix)
        """
        j = self.__get_random_item()
        while j == item or (user, j) in container:
            j = self.__get_random_item()
        return j

    def get_negative_sampling(self, container: Container, num: int=1) -> np.ndarray:
        """
        create negative samples for the dataset interactions
        with random item ids that don't match existing interactions

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param num: amount of samples per interaction
        :return array of dimensions (self.shape[0]*num, self.shape[1])
        """

        shape = self.__interactions.shape
        data = np.zeros((shape[0]*num, shape[1]), int)
        i = 0
        for row in self.__interactions:
            user, item = row[:2]
            for _ in range(num):
                nitem = self.get_random_negative_item(user, item, container)
                data[i][:2] = (user, nitem) 
                i += 1
        return data

    def add_negative_sampling(self, container: Container, num: int=1, group: bool=False) -> None:
        """
        add negative samples to the dataset interactions
        with random item ids that don't match existing interactions

        :param container: container to check if the interaction exists (usually the adjacency matrix)
        :param num: amount of samples per interaction
        :param group: if the interaction + negative samples should be grouped (easier for test datasets)
        """
        if num <= 0:
            return
        samples = self.get_negative_sampling(container, num)
        if group:
            pass
        else:
            self.__interactions = np.r_[self.__interactions, samples]

    def create_adjacency_matrix(self) -> sp.dok_matrix:
        """
        create the adjacency matrix for the dataset
        """
        size = self.idsize[1]
        matrix = sp.dok_matrix((size, size), dtype=np.float32)
        for row in self.__interactions:
            user, item = row[:2]
            matrix[user, item] = 1.0
            matrix[item, user] = 1.0
        return matrix

    @classmethod
    def fromcsv(cls, *args, **kwargs) -> 'Dataset':
        """
        load a dataset from a CSV file using pandas
        """
        return cls(pd.read_csv(*args, **kwargs).to_numpy())

