"""
Helper functions for control module.
"""


def t2n(x):
    """
    Transform a torch tensor to a numpy array.

    Args:
        x: torch tensor

    Returns:
        (np.array): numpy array

    """
    return x.detach().cpu().numpy()
