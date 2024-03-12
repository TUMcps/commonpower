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


class ArgsWrapper:
    """
    Allows us to access dictionary items by calling dict.key instead of dict["key"]
    """

    def __init__(self, args_dict):
        self.__dict__ = args_dict

    def __getattr__(self, name):
        return self.__dict__.get(name)
