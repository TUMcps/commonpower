"""
Collection of penalties.
"""


class BasePenalty:
    """
    Base class for penalties.
    A penalty is computed if the action proposed by the agent
    needs to be changed by the safety layer.
    """

    def get_correction_penalty(self, **kwargs) -> float:
        """
        Computes the penalty, must be implememnted by subclasses.
        args:
            None
        returns:
            float: penalty
        """
        raise NotImplementedError


class NoPenalty(BasePenalty):
    """
    No additional penalty for action correction.
    """

    def get_correction_penalty(self) -> float:
        """
        No penalty for action correction, therefore returns 0 always.
        args:
            None
        returns:
            float: penalty_value
        """
        return 0.0


class ConstantPenalty(BasePenalty):
    def __init__(self, penalty_constant: float = 1.0) -> None:
        """
        Constant penalty.
        args:
            penalty_factor (float): factor for penalty
        returns:
            None
        """
        super().__init__()
        self.penalty_constant = penalty_constant

    def get_correction_penalty(self) -> float:
        """
        Computes the penalty.
        args:
            None
        returns:
            float: penalty_value
        """
        return self.penalty_constant


class DistanceDependingPenalty(BasePenalty):
    def __init__(self, penalty_factor: float = 0.1) -> None:
        """
        Penalty depending on the distance between the proposed action
        and the corrected action.
        args:
            penalty_factor (float): factor for penalty
        returns:
            None
        """
        super().__init__()
        self.penalty_factor = penalty_factor

    def get_correction_penalty(self, safety_obj: float) -> float:
        """
        Computes the penalty.
        args:
            safety_obj (float): The optimal value of the objective function of the safety shield
        returns:
            float: penalty_value
        """
        return self.penalty_factor * safety_obj
