from nnrecommend.dataset import Dataset


def test_dataset():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    assert len(dataset) == 2
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()
    assert (dataset.idsize == (2, 4)).all()


def test_adjacency_matrix():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert (0, 3) in matrix
    assert (0, 2) not in matrix
    assert (1, 2) in matrix
    nitem = dataset.get_random_negative_item(0, 3, matrix)
    assert nitem == 2


def test_negative_sampling():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix)
    assert len(dataset) == 4
    assert (dataset[2] == (0, 2, 0)).all()
    assert (dataset[3] == (1, 3, 0)).all()