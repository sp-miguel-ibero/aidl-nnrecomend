import numpy as np
import pytest
from nnrecommend.dataset import InteractionDataset, InteractionPairDataset


def test_dataset():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
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
    dataset = InteractionDataset(data)
    mapping = dataset.normalize_ids()
    assert (mapping[0] == (2, 3, 4)).all()
    assert (mapping[1] == (1, 2)).all()


def test_dataset_denormalize():
    data = ((2, 2), (3, 1), (4, 1))
    dataset = InteractionDataset(data)
    mapping = dataset.normalize_ids()
    dataset.denormalize_ids(mapping)
    assert (dataset[0] == (2, 2, 1)).all()
    assert (dataset[1] == (3, 1, 1)).all()
    assert (dataset[2] == (4, 1, 1)).all()


def test_dataset_pass_mapping():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    mapping = dataset.map_ids(((2, 3),(1, 2)))
    assert (mapping[0] == (2, 3)).all()
    assert (mapping[1] == (1, 2)).all()
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()


def test_dataset_bad_mapping():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    dataset.map_ids(((1, 2),(1, 2)))
    assert len(dataset) == 1
    assert (dataset[0] == (1, 3, 1)).all()


def test_adjacency_matrix():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert (0, 3) in matrix
    assert (0, 2) not in matrix
    assert (1, 2) in matrix
    assert (dataset[0] == (0, 3, 1)).all()
    nitems = dataset.get_random_negative_rows(dataset[0], 3, matrix)
    assert nitems.shape[0] == 3
    assert (nitems[0] == (0, 2, 0)).all()
    assert (nitems[1] == (0, 2, 0)).all()
    assert (nitems[2] == (0, 2, 0)).all()


def test_add_negative_sampling():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(2, matrix)
    assert len(dataset) == 6
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()
    assert (dataset[2] == (0, 2, 0)).all()
    assert (dataset[3] == (0, 2, 0)).all()
    assert (dataset[4] == (1, 3, 0)).all()
    assert (dataset[5] == (1, 3, 0)).all()


def test_add_unique_negative_sampling():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(1, matrix, unique=True)
    assert len(dataset) == 4
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()
    assert (dataset[2] == (0, 2, 0)).all()
    assert (dataset[3] == (1, 3, 0)).all()
    with pytest.raises(ValueError):
        # not enough values
        dataset.add_negative_sampling(2, matrix, unique=True)

def test_complete_negative_sampling():
    data = ((2, 2), (3, 1), (4, 3), (4, 2))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    indices = dataset.add_negative_sampling(-1, matrix)
    assert len(dataset) == 10
    assert (dataset[0] == (0, 4, 1)).all()
    assert (dataset[1] == (1, 3, 1)).all()
    assert (dataset[2] == (2, 5, 1)).all()
    assert (dataset[3] == (2, 4, 1)).all()
    assert (dataset[4] == (0, 3, 0)).all()
    assert (dataset[5] == (0, 5, 0)).all()
    assert (dataset[6] == (1, 4, 0)).all()
    assert (dataset[7] == (1, 5, 0)).all()
    assert (dataset[8] == (2, 3, 0)).all()
    assert (dataset[9] == (2, 3, 0)).all()
    assert len(indices) == 4
    assert (indices[0] == (0, 4, 5)).all()
    assert (indices[1] == (1, 6, 7)).all()
    assert (indices[2] == (2, 8)).all()
    assert (indices[3] == (3, 9)).all()

@pytest.mark.parametrize("size", (10, 50, 100, 500))
def test_get_unique_random_negative_items(size):
    data = ((0, 0), (0, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.idrange[1] = size + 3
    items = dataset.get_unique_random_negative_items(0, 0, size, matrix)
    assert 0 not in items
    assert 1 not in items
    assert len(items) == size
    assert len(np.unique(items)) == size
    

def test_extract_negative_dataset():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(2, matrix)
    dataset2 = dataset.extract_negative_dataset()
    assert len(dataset) == 2
    assert len(dataset2) == 4
    assert (dataset2[0] == (0, 2, 0)).all()
    assert (dataset2[1] == (0, 2, 0)).all()
    assert (dataset2[2] == (1, 3, 0)).all()
    assert (dataset2[3] == (1, 3, 0)).all()


def test_extract_test_dataset():
    data = ((2, 2), (2, 3), (3, 1), (3, 4))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(2, matrix)
    assert len(dataset) == 12
    testset = dataset.extract_test_dataset()
    assert type(testset) == InteractionDataset
    assert len(dataset) == 10
    assert len(testset) == 2


def test_remove_low():
    data = ((2, 2), (2, 3), (3, 1), (3, 4), (4, 1))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert len(dataset) == 5
    dataset.remove_low_users(matrix, 1)
    assert len(dataset) == 4
    dataset.remove_low_items(matrix, 1)
    assert len(dataset) == 1


def test_dataset_context():
    # last column is the label
    data = ((2, 20, 2, 0), (3, 20, 0, 1), (2, 20, 7, 0.5))
    dataset = InteractionDataset(data)
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
    dataset = InteractionDataset(data)
    mapping = dataset.normalize_ids()
    dataset.denormalize_ids(mapping)
    assert len(dataset) == n
    for i in range(len(dataset)):
        assert (dataset[i] == data[i]).all()


def test_add_prev_item():
    data = ((2, 2), (2, 3), (3, 4), (4, 1), (3, 4))
    dataset = InteractionDataset(data)
    dataset.add_previous_item_column()
    assert dataset[0].shape[0] == 4
    assert (dataset.idrange == (3, 7, 12)).all()
    assert (dataset[0] == (0, 4, 7, 1)).all()
    assert (dataset[1] == (0, 5, 9, 1)).all()
    assert (dataset[2] == (1, 6, 7, 1)).all()
    assert (dataset[3] == (2, 3, 7, 1)).all()
    assert (dataset[4] == (1, 6, 11, 1)).all()


def test_pair_dataset():
    data = ((2, 2), (3, 1), (4, 3), (4, 2))
    dataset = InteractionDataset(data)
    matrix = dataset.create_adjacency_matrix()
    indices = dataset.add_negative_sampling(-1, matrix)
    dataset = InteractionPairDataset(dataset, indices)
    assert len(dataset) == 6
    assert (dataset[0][0] == (0, 4, 1)).all()
    assert (dataset[0][1] == (0, 3, 0)).all()
    assert (dataset[1][0] == (0, 4, 1)).all()
    assert (dataset[1][1] == (0, 5, 0)).all()

    assert (dataset[2][0] == (1, 3, 1)).all()
    assert (dataset[2][1] == (1, 4, 0)).all()
    assert (dataset[3][0] == (1, 3, 1)).all()
    assert (dataset[3][1] == (1, 5, 0)).all()

    assert (dataset[4][0] == (2, 5, 1)).all()
    assert (dataset[4][1] == (2, 3, 0)).all()
    assert (dataset[5][0] == (2, 4, 1)).all()
    assert (dataset[5][1] == (2, 3, 0)).all()


def test_combine_columns():
    data = ((2, 2, 1, 1), (3, 1, 5, 1))
    dataset = InteractionDataset(data)
    dataset.combine_columns(1, 2)
    assert len(dataset) == 2
    assert len(dataset.idrange) == 2

    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 4, 1)).all()
    assert dataset.idrange[0] == 2
    assert dataset.idrange[1] == 6


def test_remove_column():
    data = ((2, 2), (2, 1), (1, 3), (1, 2))
    dataset = InteractionDataset(data)
    dataset.normalize_ids()
    dataset.remove_column(0)
    assert len(dataset) == 4
    assert len(dataset.idrange) == 1
    assert dataset.idrange[0] == 3
    assert (dataset[0] == (1, 1)).all()
    assert (dataset[1] == (0, 1)).all()
    assert (dataset[2] == (2, 1)).all()
    assert (dataset[3] == (1, 1)).all()
    dataset.remove_column(0)
    assert len(dataset) == 4
    assert len(dataset.idrange) == 0
    assert (dataset[0] == (1,)).all()
    assert (dataset[1] == (1,)).all()
    assert (dataset[2] == (1,)).all()
    assert (dataset[3] == (1,)).all()


def test_prepare_for_recommend():
    data = ((2, 2), (2, 1), (1, 3), (1, 2))
    dataset = InteractionDataset(data)
    dataset.add_previous_item_column()
    dataset.unify_column(0)
    items = dataset[:, 1] - 1
    dataset.remove_column(1)
    dataset[:, -1] = items

    assert len(dataset) == 4
    assert len(dataset.idrange) == 2
    assert (dataset.idrange == (1, 5)).all()
    assert (dataset[0] == (0, 1, 1)).all()
    assert (dataset[1] == (0, 3, 0)).all()
    assert (dataset[2] == (0, 1, 2)).all()
    assert (dataset[3] == (0, 4, 1)).all()


def test_counts():
    data = ((2, 2), (2, 1), (2, 2), (1, 2), (2, 1), (2, 1))
    dataset = InteractionDataset(data)
    counts = dataset.get_counts()
    assert (counts == (2, 3, 2, 1, 3, 3)).all()

def test_swap_columns():
    data = ((2, 2), (3, 1))
    dataset = InteractionDataset(data)
    dataset.swap_columns(0, 1)
    assert len(dataset) == 2
    assert (dataset[0] == (1, 2, 1)).all()
    assert (dataset[1] == (0, 3, 1)).all()
    assert (dataset.idrange == (2, 4)).all()

def test_prepare_for_recommend():
    data = ((2, 2), (2, 3), (3, 4), (4, 1), (3, 4))
    dataset = InteractionDataset(data)
    dataset.normalize_ids()
    assert (dataset.idrange == (3, 7)).all()
    dataset.prepare_for_recommend()
    assert dataset[0].shape[0] == 3
    assert len(dataset) == 2
    assert (dataset.idrange == (4, 8)).all()
    assert (dataset[0] == (1, 6, 1)).all()
    assert (dataset[1] == (3, 7, 1)).all()
