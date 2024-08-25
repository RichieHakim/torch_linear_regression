import numpy as np
import torch

def check_full_rank(X, a_thresh=None, r_thresh=None):
    """
    Check if a matrix is full rank.

    Args:
        X (torch.Tensor): 
            The matrix to check.
            Should be a 2D numpy array or torch.Tensor.
        a_thresh (float, optional):
            The absolute threshold for the singular values; singular values
            below this threshold are considered zero.
            If None, then not used.
        r_thresh (float, optional):
            The relative threshold for the singular values; singular values
            below this threshold are considered zero.
            Here, 'relative' means relative to the largest singular value.

    Returns:
        bool: 
            True if the matrix is rank-deficient, False otherwise.
    """
    U, S, V = torch.linalg.svd(torch.as_tensor(X))

    S_bool = torch.ones_like(S, dtype=bool)
    if a_thresh is not None:
        S_bool = S_bool & (S > a_thresh)
    if r_thresh is not None:
        S_bool = S_bool & (S > r_thresh * S[0])

    return not torch.all(S_bool)
    
