from torch.utils.data import Dataset,Subset
import numpy as np

def split_dataset(dataset: Dataset | Subset,
                  length: int = None, percent: float = None,
                  shuffle: bool = True
                  ) -> tuple[Subset, Subset]:
    r"""Split a dataset into two subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        length (int): The length of the first subset.
            This argument cannot be used together with :attr:`percent`.
            If ``None``, use :attr:`percent` to calculate length instead.
            Defaults to ``None``.
        percent (float): The split ratio for the first subset.
            This argument cannot be used together with :attr:`length`.
            ``length = percent * len(dataset)``.
            Defaults to ``None``.
        shuffle (bool): Whether to shuffle the dataset.
            Defaults to ``True``.
        seed (bool): The random seed to split dataset
            using :any:`numpy.random.shuffle`.
            Defaults to ``None``.

    Returns:
        (torch.utils.data.Subset, torch.utils.data.Subset):
            The two splitted subsets.

    :Example:
        >>> import torch
        >>> from trojanzoo.utils.data import TensorListDataset, split_dataset
        >>>
        >>> data = torch.ones(11, 3, 32, 32)
        >>> targets = list(range(11))
        >>> dataset = TensorListDataset(data, targets)
        >>> set1, set2 = split_dataset(dataset, length=3)
        >>> len(set1), len(set2)
        (3, 8)
        >>> set3, set4 = split_dataset(dataset, percent=0.5)
        >>> len(set3), len(set4)
        (5, 6)

    Note:
        This is the implementation of :meth:`trojanzoo.datasets.Dataset.split_dataset`.
        The difference is that this method will NOT set :attr:`seed`
        as ``env['data_seed']`` when it is ``None``.
    """
    assert (length is None) != (percent is None)  # XOR check
    length = length 
    #TODO: if batch_size != 64
    if length % 64 != 0:
        length += 64 - (length % 64)
        if length > len(dataset):
            length -= 64
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    if isinstance(dataset, Subset):
        idx = np.array(dataset.indices)
        indices = idx[indices]
        dataset = dataset.dataset
    subset1 = Subset(dataset, indices[:length])
    subset2 = Subset(dataset, indices[length:])
    return subset1, subset2