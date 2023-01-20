import torch
from torch import nn


class AffinityLoss:
    """Affinity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : np.ndarray (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The affinity similarity should be normalized to the range (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor):
        self.lookup = torch.tensor(lookup)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = nn.L1Loss()

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, device
    ) -> torch.Tensor:
        """Return the affinity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the object as an index. These indices should
            correspond to the rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.

        Returns
        -------
        loss : torch.Tensor
            The affinity loss.
        """
        # first calculate the affinity ,for the real classes
        c = torch.combinations(y_true, r=2, with_replacement=False).to(device)
        affinity = self.lookup[c[:, 0], c[:, 1]].to(device)

        # now calculate the latent similarity
        z_id = torch.tensor(list(range(y_pred.shape[0])))
        c = torch.combinations(z_id, r=2, with_replacement=False)
        latent_similarity = self.cos(y_pred[c[:, 0], :], y_pred[c[:, 1], :])
        loss = self.l1loss(latent_similarity, affinity)
        return loss
