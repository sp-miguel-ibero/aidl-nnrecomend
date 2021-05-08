
import os
import numpy as np
from nnrecommend.dataset import Dataset


DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../datasets/ml-dataset-splitted/movielens")


def test_dataset():
    dataset = Dataset.fromcsv(f"{DATASET_PATH}.train.rating", sep='\t', header=None)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, 4)
    testset = Dataset.fromcsv(f"{DATASET_PATH}.test.rating", sep='\t', header=None)
    testset.add_negative_sampling(matrix, 99, True)

    assert len(dataset) == 5*99057
    assert (dataset.idsize == (943, 2625)).all()
    assert matrix.shape == (2625, 2625)
    assert len(testset) == 943
    assert (dataset[50] == (0, 113+943, 1, 875072173)).all()
    assert (dataset[90000] == (861, 1008+943, 1, 879303622)).all()
    assert (testset[500] == (500, 1949, 1, 883995203)).all()
