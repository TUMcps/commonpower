from pyomo.core import ConcreteModel
from pyomo.core.base.indexed_component import IndexedComponent

from commonpower.utils.helpers import rgetattr, rhasattr


def model_root(model: ConcreteModel) -> ConcreteModel:
    """
    Returns the root model of the given model by recursively calling model.parent_block().

    Args:
        model (ConcreteModel): Model.

    Returns:
        ConcreteModel: Root model.
    """

    def get_root(model: ConcreteModel) -> ConcreteModel:
        root = model
        parent = model.parent_block()
        if parent is not None:
            root = get_root(parent)
        return root

    return get_root(model)


def get_element_from_model(name: str, model: ConcreteModel, local_id: str, global_id: str) -> IndexedComponent:

    root_model = model_root(model)

    # Try global access (works if root_model == global model)
    if rhasattr(root_model, global_id):
        return rgetattr(root_model, global_id)

    # Try local access if a local element was passed and the passed model is its own root model.
    # This would only happen for the system block or "cut-off" sub-global blocks (e.g. as accessed by controller)
    # that want to access a top-level element
    if model == root_model and local_id == name and rhasattr(model, local_id):
        return rgetattr(model, local_id)

    # Usually, the passed model has a root model higher up the hierarchy.
    # If global access did not work, this root is not the global model
    # (e.g. if the root is a sub-global block from a controller).
    # This is why we iterate though the global element id top-down until we find the right element.
    # We already know that local access did not work, so we will not try the local id.
    # This prevents finding the wrong element if it exists on a higher level.
    # E.g. "n0.n01.e1.p" -> "n01.e1.p" -> "e1.p" !-> "p"
    for level in range(len(global_id.split(".")) - 1):
        name_for_level = ".".join(global_id.split(".")[level:])
        if rhasattr(root_model, name_for_level):
            return rgetattr(root_model, name_for_level)


class SubscriptableFloat(float):
    """
    This is a dummy class to "fake" the behaviour of a model element
    when extracting the signature from a constraint/cost expression.
    """

    def __getitem__(self, _):
        """Make the float subscriptable by returning itself for any index"""
        return self

    def is_indexed(self):
        return False

    @property
    def ub(self):
        return self

    @property
    def lb(self):
        return self
