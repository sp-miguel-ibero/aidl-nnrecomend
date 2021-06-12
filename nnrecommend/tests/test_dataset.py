import numpy as np
import pytest
from nnrecommend.dataset import Dataset


def test_dataset():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    assert len(dataset) == 2
    assert (dataset[0] == (2, 2, 1)).all()
    assert (dataset[1] == (3, 1, 1)).all()
    mapping = dataset.normalize_ids()
    assert len(dataset) == 2
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()
    assert (dataset.idrange == (2, 4)).all()
    assert (mapping[0] == (2, 3)).all()
    assert (mapping[1] == (1, 2)).all()


def test_dataset_mapping():
    data = ((2, 2), (3, 1), (4, 1))
    dataset = Dataset(data)
    mapping = dataset.normalize_ids()
    assert (mapping[0] == (2, 3, 4)).all()
    assert (mapping[1] == (1, 2)).all()


def test_dataset_denormalize():
    data = ((2, 2), (3, 1), (4, 1))
    dataset = Dataset(data)
    mapping = dataset.normalize_ids()
    dataset.denormalize_ids(mapping)
    assert (dataset[0] == (2, 2, 1)).all()
    assert (dataset[1] == (3, 1, 1)).all()
    assert (dataset[2] == (4, 1, 1)).all()


def test_dataset_pass_mapping():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    mapping = dataset.map_ids(((2, 3),(1, 2)))
    assert (mapping[0] == (2, 3)).all()
    assert (mapping[1] == (1, 2)).all()
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()


def test_dataset_bad_mapping():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    dataset.map_ids(((1, 2),(1, 2)), remove_missing=False)
    assert (dataset[0] == (1, 3, 1)).all()
    assert (dataset[1] == (-1, 2, 1)).all()


def test_adjacency_matrix():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert (0, 3) in matrix
    assert (0, 2) not in matrix
    assert (1, 2) in matrix
    assert (dataset[0] == (0, 3, 1)).all()
    nitems = dataset.get_random_negative_rows(matrix, dataset[0], 3)
    assert nitems.shape[0] == 3
    assert (nitems[0] == (0, 2, 0)).all()
    assert (nitems[1] == (0, 2, 0)).all()
    assert (nitems[2] == (0, 2, 0)).all()


def test_negative_sampling():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, 2)
    assert len(dataset) == 6
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (0, 2, 0)).all()
    assert (dataset[2] == (0, 2, 0)).all()
    assert (dataset[3] == (1, 2, 1)).all()
    assert (dataset[4] == (1, 3, 0)).all()
    assert (dataset[5] == (1, 3, 0)).all()


def test_extract_test_dataset():
    data = ((2, 2), (2, 3), (3, 1), (3, 4))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, 2)
    assert len(dataset) == 12
    testset = dataset.extract_test_dataset()
    assert type(testset) == Dataset
    assert len(dataset) == 10
    assert len(testset) == 2


def test_remove_low():
    data = ((2, 2), (2, 3), (3, 1), (3, 4), (4, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert len(dataset) == 5
    dataset.remove_low_users(matrix, 1)
    assert len(dataset) == 4
    dataset.remove_low_items(matrix, 1)
    assert len(dataset) == 1


def test_dataset_context():
    # last column is the label
    data = ((2, 20, 2, 0), (3, 20, 0, 1), (2, 20, 7, 0.5))
    dataset = Dataset(data)
    assert len(dataset) == 3
    assert (dataset[0] == (2, 20, 2, 0)).all()
    assert (dataset[1] == (3, 20, 0, 1)).all()
    assert (dataset[2] == (2, 20, 7, 0.5)).all()
    mapping = dataset.normalize_ids()
    assert len(mapping) == 3
    assert (mapping[0] == (2, 3)).all()
    assert (mapping[1] == (20)).all()
    assert (mapping[2] == (0, 2, 7)).all()
    assert (dataset.idrange == (2, 3, 6)).all()
    assert len(dataset) == 3
    assert (dataset[0] == (0, 2, 4, 0)).all()
    assert (dataset[1] == (1, 2, 3, 1)).all()
    assert (dataset[2] == (0, 2, 5, 0.5)).all()


@pytest.mark.parametrize("n, s, l", [(10, 10, 100)])
def test_dataset_context_denormalize(n, s, l):
    data = []
    for i in range(n):
        data.append(np.random.randint(0, l, size=s))
    dataset = Dataset(data)
    mapping = dataset.normalize_ids()
    dataset.denormalize_ids(mapping)
    assert len(dataset) == n
    for i in range(len(dataset)):
        assert (dataset[i] == data[i]).all()