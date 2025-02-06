"""
Generic abstractions and functionality for interacting with the pyomo layer.
"""
from __future__ import annotations

import json
import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, Union

import gymnasium as gym
import numpy as np
import pyomo.environ as pyo
from prettytable import PrettyTable
from pyomo.core import Block, ConcreteModel, Constraint, Expression, Objective, Param, Set, Var

from commonpower.data_forecasting.base import DataProvider
from commonpower.modeling.param_initialization import ParamInitializer
from commonpower.modeling.robust_cost import CostScenario
from commonpower.modeling.util import get_element_from_model
from commonpower.utils import rgetattr, rsetattr
from commonpower.utils.cp_exceptions import EntityError

if TYPE_CHECKING:
    pass


class ElementTypes(IntEnum):
    """
    The ElementTypes describe the type of ModelElements.
    This is necessary to allow for specific treatment.
    """

    #: State variable. Maps to the Pyomo type Var.
    STATE = 1
    #: Generic variable. The difference to state variables is that VAR does not have to be initialized.
    #: Maps to the Pyomo type Var.
    VAR = 2
    #: Input variable. Maps to the Pyomo type Var.
    INPUT = 3
    #: Exogenous input, which is read from a data provider. Maps to the Pyomo type Param.
    DATA = 4
    #: Fixed Parameter. Parameters can either be constant across runs or be initialized in each run
    #: based on a specific logic. Maps to the Pyomo type Param.
    CONSTANT = 5
    #: Constraint. Input coupling and dynamics functions are defined with this type.
    #: Maps to the Pyomo type Constraint.
    CONSTRAINT = 6
    #: Robust constraint. This is a special type of constraint that is evaluated by the RobustConstraintBuilder.
    #: Is expanded to (usually) multiple instances of the Pyomo type Constraint.
    ROBUST_CONSTRAINT = 7
    #: Cost variable. This is essentially a generic variable but explicitly defined to simplify
    #: downstream analysis. Maps to the Pyomo type Var.
    COST = 8
    #: Set. Sets can be useful to specify the values a discrete variable can take. Maps to the Pyomo type Set.
    SET = 9


class ModelElement:
    def __init__(
        self,
        name: str,
        element_type: ElementTypes,
        doc: str,
        domain: Union[pyo.Integers, pyo.Binary, pyo.Reals] = pyo.Reals,
        bounds: Union[None, tuple[float]] = None,
        expr: Union[None, callable] = None,
        initialize: Union[None, any] = None,
        indexed: Union[None, bool] = None,
        uncertainty_bounds: Union[None, tuple[float]] = None,
    ) -> ModelElement:
        """
        The ModelElement class builds the bridge between the CommonPower object space
        and the pyomo model representation.
        Since pyomo does not give us a lot of nuance wrt. the "meaning" of variables/parameters,
        we capture that via the ElementTypes.
        We then map the ElementTypes to corresponding pyomo classes and provide an interface to
        instantiate and add them to the pyomo model.

        Args:
            name (str): Name of the model element.
            element_type (ElementTypes): Element type. Will be mapped to the appropriate pyomo class.
            doc (str): Additional info/description. Will be passed to the pyomo class as the "doc" argument.
            domain (Union[pyo.Integers, pyo.Binary, pyo.Reals], optional): Pyomo domain. Defaults to pyo.Reals.
            bounds (Union[None, tuple[float]], optional): Lower and upper bounds. Can be overwritten in add_to_model().
                Defaults to None.
            expr (Union[None, callable], optional): Expression for constraints. Defaults to None.
            initialize (Union[None, any], optional): Values to initialize the element with.
            indexed (Union[None, bool], optional): Specifies if the variable should be indexed on model.t.
                If not provided, all elements except CONSTANT/SET are indexed.
                Currently the indexing of Constraints and non-indexing of Sets are enforced.
                TODO: Fully implement indexing flexibility.
            uncertainty_bounds (Union[None, tuple[float]], optional): Interval bounds for the uncertainty set.
                Only relevant for ElementTypes.CONSTANT. Defaults to None.

        Raises:
            AttributeError: If the given type is unknown or if required arguments are not provided.
        """
        self.name = name
        self.type = element_type
        self.doc = doc
        self.expr = expr
        self.domain = domain
        self.bounds = bounds
        self.initialize = initialize
        self.indexed = indexed if indexed is not None else True
        self.uncertainty_bounds = tuple(uncertainty_bounds) if uncertainty_bounds is not None else None

        # autogenerate bounds for Binary domain (this way we do not try to look for bounds in the config dict)
        if domain == pyo.Binary:
            self.bounds = (0, 1)

        # mapping to pyomo types
        if self.type == ElementTypes.STATE:
            self.pyomo_class = Var
        elif self.type == ElementTypes.VAR:
            self.pyomo_class = Var
        elif self.type == ElementTypes.INPUT:
            self.pyomo_class = Var
        elif self.type == ElementTypes.COST:
            self.pyomo_class = Var
        elif self.type == ElementTypes.DATA:
            self.pyomo_class = Param
        elif self.type == ElementTypes.CONSTANT:
            self.pyomo_class = Param
            self.indexed = indexed if indexed is not None else False
        elif self.type == ElementTypes.SET:
            self.pyomo_class = Set
            self.indexed = indexed if indexed is not None else False
            if not initialize:
                raise AttributeError("No init for set specified")
        elif self.type in [ElementTypes.CONSTRAINT, ElementTypes.ROBUST_CONSTRAINT]:
            self.pyomo_class = Constraint
            if not expr:
                raise AttributeError("No expr for constraint specified")
        else:
            raise AttributeError(f"Unknown element type: {self.type}")

    def add_to_model(
        self,
        model: ConcreteModel,
        name: str,
        bounds: Union[None, tuple[float]] = None,
        initialize: Union[None, int, float, ParamInitializer] = None,
    ) -> None:
        """
        Here we parse the ModelElement to the corresponding pyomo model element and add it to the given model.
        Some assumptions are made:
            - All elements mapping to Var/Constraint are automatically indexed with "model.t".
            - All elements mapping to Var are initialized at the middle between their lower and upper bounds.
            - All elements mapping to Param are defined as mutable.

        Args:
            model (ConcreteModel): Pyomo model to add the element to.
            name (str): Complete name that the element should have in the pyomo model.
                This is not the same as self.name because it depends on the scope of the given model.
            bounds (Union[None, tuple[float]], optional): Lower and upper bounds.
                Overwrite self.bounds if given. Defaults to None.
            initialize (Union[None, int, float, ParamInitializer], optional): Only relevant for ElementTypes
                that are mapped to Param: Value to initialize the pyomo element with.
                If self.initialize was defined, we ignore whatever is passed here.
                If the argument (or self.initialize) is neither int nor float, the Param will be initialized at zero.
                Defaults to None.

        Raises:
            NotImplementedError: If no mapping exists for self.pyomo_class.
        """
        if self.pyomo_class == Constraint:
            if self.indexed is True:
                pyomo_el = self.pyomo_class(model.t, doc=self.doc, expr=self.expr)
            else:
                pyomo_el = self.pyomo_class(doc=self.doc, expr=self.expr)

        elif self.pyomo_class == Var:
            if bounds:
                if self.bounds:
                    logging.debug(f"Overriding default bounds {self.bounds} on model element {name} with {bounds}")
                self.bounds = bounds

            if self.bounds:
                if self.indexed is True:

                    def var_init(model, t):  # set the initial value of variables to the center between their bounds
                        if self.domain == pyo.Binary:
                            return 0
                        else:
                            return (self.bounds[0] + self.bounds[1]) / 2

                    pyomo_el = self.pyomo_class(
                        model.t,
                        doc=self.doc,
                        initialize=var_init,
                        bounds=(self.bounds[0], self.bounds[1]),
                        domain=self.domain,
                    )

                else:

                    def var_init(model):  # set the initial value of variables to the center between their bounds
                        if self.domain == pyo.Binary:
                            return 0
                        else:
                            return (self.bounds[0] + self.bounds[1]) / 2

                    pyomo_el = self.pyomo_class(
                        doc=self.doc, initialize=var_init, bounds=(self.bounds[0], self.bounds[1]), domain=self.domain
                    )

            else:  # does this ever happen?

                def zero_init(model, t):
                    return 0

                if self.indexed is True:
                    pyomo_el = self.pyomo_class(model.t, doc=self.doc, initialize=zero_init, domain=self.domain)
                else:
                    pyomo_el = self.pyomo_class(doc=self.doc, initialize=zero_init, domain=self.domain)

        elif self.pyomo_class == Param:
            # maybe something was defined already
            initialize = self.initialize if self.initialize is not None else initialize

            if not isinstance(initialize, (int, float)):  # it might be of type ParamInitializer
                initialize = 0

            # Params can be indexed if they are e.g. coming from DataSources
            if self.indexed is True:
                pyomo_el = self.pyomo_class(
                    model.t, doc=self.doc, mutable=True, initialize=initialize, domain=self.domain
                )
            else:
                pyomo_el = self.pyomo_class(doc=self.doc, mutable=True, initialize=initialize, domain=self.domain)

        elif self.pyomo_class == Set:
            pyomo_el = self.pyomo_class(initialize=self.initialize)

        else:
            raise NotImplementedError(f"ModelElement {self.name} has unsupported pyomo class: {self.pyomo_class}")

        rsetattr(model, name, pyomo_el)


class ModelEntity:
    @classmethod
    def info(cls) -> None:
        """
        Prints some information about this entity.
        Included are ModelEntities with the corresponding configurations and data providers.
        """
        model_elements = cls._augment_model_elements(cls._get_model_elements())

        print(f"\n---- INFO: {cls.__name__} ----\n")
        # print(f"\nMODEL ELEMENTS:\n")

        config_template = {}

        tab = PrettyTable(["Element", "Type", "Description", "Domain", "Bounds", "Required config", "Data provider"])

        for el in model_elements:
            req_config = ""
            req_dp = ""
            if el.type == ElementTypes.CONSTANT and el.initialize is None:
                # "constants" can be defined either by a constant float or a ParamInitializer which is called on reset()
                # except for state_inits, which require a ParamInitializer
                if "_init" in el.name:
                    req_config = "ParamInitializer"
                else:
                    req_config = "constant or ParamInitializer"
            elif el.type in [ElementTypes.INPUT, ElementTypes.VAR, ElementTypes.STATE]:
                if not el.bounds:
                    req_config = "(lb, ub)"
            elif el.type == ElementTypes.DATA:
                req_dp = "Yes"

            if req_config:
                config_template[el.name] = req_config + f" ({el.domain})"

            tab.add_row([el.name, ElementTypes(el.type).name, el.doc, el.domain, el.bounds, req_config, req_dp])

        print(tab)

        print("\nCONFIG TEMPLATE\n")

        print(json.dumps(config_template, indent=4))

        print("\n---- INFO END ----\n")

    @classmethod
    def _get_model_elements(cls) -> list[ModelElement]:
        """
        This is the central method which all subclasses must implement.
        Here, the model elements of the entity are defined.
        For clarity, specify main variables and parameters here and specify constraints and
        auxiliary variables in _augment_model_elements().

        Returns:
            list[ModelElement]: List of model elements which will represent the entity in the pyomo model.
        """
        raise NotImplementedError

    @classmethod
    def _augment_model_elements(cls, model_elements: list[ModelElement]) -> list[ModelElement]:
        """
        This method augments the list of model elements. It might add initial state variables, cost variables etc.
        Its purpose is to decouple a "leaf" object's model elements (retrieved from ._get_model_elements())
        from generic elements inherited by its parent class.
        It does not need to be implemented by subclasses.

        Args:
            model_elements (list[ModelElement]): List of main variables and parameters.

        Returns:
            list[ModelElement]: List of the given model_elements augmented by additional elements.
        """
        return model_elements

    def __init__(self, name: str, config: dict = {}) -> ModelEntity:
        """
        This class abstracts power system entities which have a pyomo model representation.
        It also bundles all interfaces needed to interact with their model.
        Subclasses of ModelEntity implement certain methods which specify the model elements associated to
        instances of that class.

        Args:
            name (str): Descriptive name of the entity. It will not be used within the pyomo model and
                is merely for human interpretability.
            config (dict, optional): Configuration dict of the entity. The content required depends on the modelling of
                the specific subclass. Defaults to {}.
        """
        self.model = None
        self.instance = None
        self.controller = None

        self.name = name
        self.id = ""

        self.model_elements: list[ModelElement] = []
        self.data_providers: list[DataProvider] = []
        self.data_provider_map: dict[str, DataProvider] = {}  # maps elements to their data provider

        self.config = config

    def add_to_model(self, model: ConcreteModel, **kwargs) -> None:
        """
        This method adds the calling entity to the given (global) pyomo model.
        To this end, we
            - declare and add a new pyomo block named by self.id (the entity's global id).
            - call _get_model_elements() to retrieve the entity's model elements (variables and parameters).
            - call _augment_model_elements() to add additional model elements (constraints etc.).
            - check the configuration dict for completeness based on the defined model elements.
            - add all model elements to the previously declared pyomo block.

        We also store a reference to the global model in self.model.

        Args:
            model (ConcreteModel): Global pyomo model.
        """
        self.model = model  # store reference to global model internally
        for k, v in kwargs:
            setattr(self, k, v)

        rsetattr(self.model, self.id, ConcreteModel())

        self.model_elements = self._augment_model_elements(self._get_model_elements())

        self.model_elements = self._add_constraints(self.model_elements)

        self._check_config(self.config)

        for el in self.model_elements:
            self._add_model_element(el)

    def add_data_provider(self, data_provider: DataProvider) -> ModelEntity:
        """
        Adds a data provider to the entity.
        It will be checked during validation if all model elements which require a data provider are covered.

        Args:
            data_provider (DataProvider): Data provider instance.

        Returns:
            ModelEntity: ModelEntity instance.
        """
        self.data_providers.append(data_provider)
        return self

    def clear_data_providers(self):
        self.data_providers = []

    def get_pyomo_element(self, name: str, model: ConcreteModel) -> Union[Var, Param, Set, Constraint, Objective]:
        """
        Gets a pyomo element referenced by name from the given model.
        The name can be local (e.g. "p", i.e. from the perspective of the calling block) or
        non-local (e.g. "n1.n12.p", i.e. from the perspective of a higher block).
        The given model can also be local (of the calling block) or of a block higher in the hierarchy.
        We first get the root of the passed model and constuct the element id for the global model.
        This will find the correct element if the passed model is the global model.
        For any sub-global model, we iteratively make the element id "more local" until we find the right element.
        We ensure that the element is on the model branch of the calling entity,
        i.e., one cannot access elements of other entities.

        Args:
            name (str): Name of the model element (can be local or global).
            model (ConcreteModel): Model to get the variable from.

        Raises:
            EntityError: If element not found.

        Returns:
            Union[Var, Param, Set, Constraint, Objective]: The referenced variable from the given model.
        """

        local_id = name.split(".")[-1] if self.id else name  # get local element id (do nothing if self is system)
        global_id = self.get_pyomo_element_id(local_id)  # get gobal element id

        # Check if name is on the branch of self.
        # This makes sure that we catch misuse, e.g., name 'n13.p' if self is 'n12'
        if name not in global_id:
            raise EntityError(self, f"The variable {name} is not on the model branch of the calling entity")

        elem = get_element_from_model(name, model, local_id, global_id)

        if elem is None:
            raise EntityError(self, f"The variable {global_id} could not be found in the given model")

        return elem

    def has_pyomo_element(self, name: str, model: ConcreteModel) -> bool:
        """
        This is essentially an indicator wrapper around get_pyomo_element() which returns False
        if no corresponding model element could be found (instead of raising an error).

        Args:
            name (str): Name of the model element (can be local or global).
            model (ConcreteModel): Model to get the variable from.

        Returns:
            bool: False if no corresponding model element could be found, True otherwise.
        """
        try:
            _ = self.get_pyomo_element(name, model)
            return True
        except EntityError:
            return False

    def get_pyomo_element_id(self, name: str) -> str:
        """
        Constructs the global element name from the local name.

        Args:
            name (str): Local element name.

        Returns:
            str: Global element name.
        """
        return self.id + "." + name if self.id else name

    def get_self_as_pyomo_block(self, model: ConcreteModel) -> Block:
        """
        Retrieves the pyomo block of the calling entity from a global model (based on the entity's global id).

        Args:
            model (ConcreteModel): Global pyomo model to access.

        Returns:
            Block: Pyomo block corresponding to the calling entity.
        """
        if not self.id:  # e.g. for System
            return model
        else:
            return rgetattr(model, self.id)

    def set_value(
        self,
        instance: ConcreteModel,
        name: str,
        val: Union[int, float, np.ndarray],
        idx: Union[None, int, list[int]] = None,
        fix_value: bool = False,
    ) -> None:
        """
        Sets the value of the specified model element to the specified value.
        Allows to specify specific indices to manipulate and to fix the variable values after setting them.

        Args:
            instance (ConcreteModel): Pyomo model to manipulate.
            name (str): Name of the element relative to the given instance (e.g. global id for global instance).
            val (Union[int, float, np.ndarray]): Value to set the element to. For indexed elements,
                an array can be passed.
            idx (Union[None, int, list[int]], optional): If only specific indices of an indexed element should be set,
                it can be specified here. If not given, it is assumed that all indices should be menipulated.
                Defaults to None.
            fix_value (bool, optional): Specifies if the values should be fixed. Defaults to False.

        Raises:
            EntityError: If an array is passed for a scalar element
                or if a scalar is passed for an indexed variable without specifying an index
                or if a list of indices is passed for a scalar element
                or if fix_value is True for an element of pyomo class Param.
        """
        el = self.get_pyomo_element(name, instance)

        if fix_value and isinstance(el, Param):
            raise EntityError(self, f"Trying to fix the value of the parameter {el.name}")

        if isinstance(val, (int, float, np.int32, np.int64, np.float32, np.float64)):
            if idx is not None:
                if isinstance(idx, list):
                    raise EntityError(
                        self, f"Setting scalar value {val} failed because multiple indices ({idx}) were provided"
                    )
                # we need this because pyomo sometimes has domain issues
                el[idx].value = round(val) if abs(round(val) - val) < 1e-8 else val
                if fix_value:
                    el[idx].fixed = True
            else:
                if el.is_indexed():
                    raise EntityError(
                        self,
                        f"Setting value {val} of indexed model element {name} failed because no index was provided",
                    )
                else:
                    el.value = round(val) if abs(round(val) - val) < 1e-8 else val
                    if fix_value:
                        el.fixed = True
        else:  # val is not a scalar
            if not el.is_indexed():
                raise EntityError(self, f"Setting value {val} of scalar model element {name} failed.")
            if idx:
                if isinstance(idx, int):
                    raise EntityError(self, f"Setting value {val} at index {idx} of model element {name} failed.")
                for i, v in enumerate(val):
                    el[idx[i]].value = round(v) if abs(round(v) - v) < 1e-8 else v
                    if fix_value:
                        el[idx[i]].fixed = True
            else:
                for i, v in enumerate(val):
                    el[i].value = round(v) if abs(round(v) - v) < 1e-8 else v
                    if fix_value:
                        el[i].fixed = True

    def get_value(self, instance: ConcreteModel, name: str) -> Union[int, float, np.ndarray]:
        """
        Gets the value of the specified model element.

        Args:
            instance (ConcreteModel): Pyomo model to access.
            name (str): Name of the element relative to the given instance (e.g. global id for global instance).

        Returns:
            Union[int, float, np.ndarray]: Value of the model element.
                If the element is indexed, we return a np.ndarray.
        """
        el = self.get_pyomo_element(name, instance)
        val = [v for v in el[:].value]
        # for non-indexed, i.e. scalar, elements we return a scalar directly
        val = np.array(val) if el.is_indexed() else val[0]
        return val

    def get_children(self) -> list[ModelEntity]:
        return []

    def cost_fcn(self, scenario: CostScenario, model: ConcreteModel, t: int = 0) -> Expression:
        """
        Returns the pyomo expression of the entity's cost function.

        Args:
            model (ConcreteModel): Model to refer to.
            t (int, optional): Time. Defaults to 0.

        Returns:
            Expression: Cost function.
        """
        return 0.0

    def _check_config(self, config: dict[str, Union[int, float]]) -> None:
        """
        Checks if all required configurations have been defined in the configuration dict passed
        to the class constructor.
        Namely, it is checked if the config contains
            - either a scalar value or an instance of ParamInitializer for all model elements of type CONSTANT.
            - bounds for all model elements of type INPUT, VAR, STATE which do not already have (default) bounds.

        Args:
            config (dict[str, Union[int, float]]): Configuration dict.

        Raises:
            EntityError: If configurations are missing and prints a list of the missing entries.
        """
        missing_elements = []
        for el in self.model_elements:
            if (
                el.type == ElementTypes.CONSTANT
            ):  # "constants" can be defined either by a constant float or a ParamInitializer
                # which is called on reset()
                if (
                    el.name not in config.keys() or not isinstance(config[el.name], (int, float, ParamInitializer))
                ) and el.initialize is None:
                    missing_elements.append((el.name, "float/int or ParamInitializer"))
            elif el.type in [ElementTypes.INPUT, ElementTypes.VAR, ElementTypes.STATE]:
                # these element types need bounds
                if (
                    el.name not in config.keys()
                    or not isinstance(config[el.name], (list, tuple))
                    or len(config[el.name]) != 2
                ) and not el.bounds:  # unless they already have bounds
                    missing_elements.append((el.name, "[lb, ub]"))

        if missing_elements:
            raise EntityError(
                self, f"The following constants have not been specified (correctly): {str(missing_elements)}"
            )

        # check if all states have corresponding initializer instances in the config
        states = [el for el in self.model_elements if el.type == ElementTypes.STATE]
        for s in states:
            if not isinstance(self.config[f"{s.name}_init"], ParamInitializer):
                raise EntityError(
                    self,
                    f"The initializer of state init parameter {s.name}_init must be of type"
                    f" {ParamInitializer.__name__}",
                )

        # check if all required dataproviders are attached
        needed_from_dataprovider = [el.name for el in self.model_elements if el.type == ElementTypes.DATA]
        if needed_from_dataprovider:
            if not self.data_providers:
                raise EntityError(self, f"Data Providers for {needed_from_dataprovider} required.")
            sourced_params = np.concatenate([s.get_variables() for s in self.data_providers], axis=None)
            if not all(x in sourced_params for x in needed_from_dataprovider):
                raise EntityError(self, f"Data Providers for {needed_from_dataprovider} required.")
            if len(set(sourced_params)) < len(sourced_params):
                raise EntityError(
                    self, f"Some variables are provided by more than one Data Provider: {sourced_params}."
                )

            # check if data sources/providers have appropriate limits
            limits_dict_el = {el.name: el.bounds for el in self.model_elements if el.type == ElementTypes.DATA}
            limits_dict_data = {
                k: v
                for limits_dict in [dp.data.get_limits() for dp in self.data_providers]
                for k, v in limits_dict.items()
            }
            for el, bounds in limits_dict_el.items():
                bounds = bounds or (-1e12, 1e12)  # el bound might be None
                if (
                    limits_dict_data[el][0] < bounds[0]
                    or limits_dict_data[el][1] > bounds[1]  # lower bound  # upper bound
                ):
                    raise EntityError(
                        self,
                        f"Data provider for {el} does not adhere to the required limits. "
                        f"Modeled limits: {bounds}, Data limits: {limits_dict_data[el]}",
                    )

        self.data_provider_map = {}
        for dp in self.data_providers:
            self.data_provider_map.update({el: dp for el in dp.get_variables()})

    def _add_model_element(self, element: ModelElement) -> None:
        """
        Adds the specified model element to self.model (by invoking element.add_to_model()).
        This method decouples ModelElements from the entity config by extracting
        configured initalization values and variable bounds.

        Args:
            element (ModelElement): Model element to add to self.model.
        """
        if element.type == ElementTypes.CONSTANT:
            element.add_to_model(
                self.model, self.get_pyomo_element_id(element.name), initialize=self.config.get(element.name, None)
            )
        elif element.type in [ElementTypes.INPUT, ElementTypes.STATE, ElementTypes.VAR]:
            element.add_to_model(
                self.model, self.get_pyomo_element_id(element.name), bounds=self.config.get(element.name, None)
            )
        else:
            element.add_to_model(self.model, self.get_pyomo_element_id(element.name))

    def _add_constraints(self, model_elements: list[ModelElement]) -> list[ModelElement]:
        """
        Adds model elements of type constraint.

        Args:
            model_elements (list[ModelElement]): Primary model elements.

        Returns:
            list[ModelElement]: Model elements augmented by constraint elements.
        """
        return model_elements


class ControllableModelEntity(ModelEntity):
    """
    This class abstracts ModelEntities which are controllable.
    """

    def register_controller(self, controller):
        """
        Register a controller with this node
        Args:
            controller (BaseController): controller to be registered

        Returns: None

        """
        self.controller = controller

    def detach_controller(self, include_children: bool = False):
        """
        Remove the current controller from the entity

        Returns: None

        """
        self.controller = None

    def n_inputs(self) -> int:
        """
        Total number of model elements with type INPUT within the entire tree of this entity

        Returns:
            int: number of inputs

        """
        n_inputs = sum([1 for e in self.model_elements if e.type == ElementTypes.INPUT])
        return n_inputs

    def input_space(self, normalize: bool = True):
        """
        Determines the input space of an entity from the bounds of all model elements with type INPUT within the tree
        Args:
            normalize (bool): Whether or not to normalize the input space to [-1,1]

        Returns:
            (None/gym.spaces.Dict): input space as a nested dictionary {element_name: box_input_space}
            in the format of the gymnasium API

        """
        # ToDo: check type of variables --> if they are binary, we cannot use box spaces?
        if self.n_inputs() == 0:
            return None
        else:
            inputs = [e for e in self.model_elements if e.type == ElementTypes.INPUT]
            lower = {}
            upper = {}
            for e in inputs:
                if e.bounds is not None:
                    if normalize:
                        lower[e.name] = -1
                        upper[e.name] = 1
                    else:
                        lower[e.name] = e.bounds[0]
                        upper[e.name] = e.bounds[1]
                else:
                    if normalize:
                        raise ValueError("Cannot normalize action space because no bounds were given for node inputs")
                    lower[e.name] = -np.inf
                    upper[e.name] = np.inf

            input_space = gym.spaces.Dict(
                {
                    e.name: gym.spaces.Box(
                        low=np.array([lower[e.name]]), high=np.array([upper[e.name]]), dtype=np.float64
                    )
                    for e in inputs
                }
            )
            return input_space

    def fix_inputs(self, inputs: Dict):
        """
        Set the variables corresponding to inputs to fixed

        Args:
            inputs: nested dictionary of inputs corresponding to model elements of type INPUT

        Returns:
            None

        """
        input_elements = [e for e in self.model_elements if e.type == ElementTypes.INPUT]
        idx = list(range(len(inputs[input_elements[0].name])))
        if len(inputs) != len(input_elements):
            raise EntityError(self, "Number of actions does not equal number of INPUT elements")
        for el in input_elements:
            el_inputs = inputs[el.name]
            self.set_value(instance=self.instance, name=el.name, val=el_inputs, idx=idx, fix_value=True)

    def get_inputs(self, model_instance: ConcreteModel = None) -> Dict:
        """
        Extracts model elements of type INPUT from a given model instance or self
        Args:
            model_instance (ConcreteModel, Optional): model to get the input elements for

        Returns:
            (None/Dict): dictionary of {element_name: array_of_input_values}

        """
        inputs = {}
        input_elements = [e for e in self.model_elements if e.type == ElementTypes.INPUT]

        if len(input_elements) == 0:
            return None

        for el in input_elements:
            if model_instance is None:
                inputs[el.name] = np.array(self.get_value(self.instance, el.name))
            else:
                inputs[el.name] = np.array(self.get_value(model_instance, el.name))
        return inputs

    def get_input_ids(self, model_instance: ConcreteModel = None) -> Union[list, None]:
        """
        Get identifiers of input elements of a given model instance or self
        Args:
            model_instance: model to get the input element identifiers for

        Returns:
            (None/list): list of identifiers of model elements of type INPUT

        """
        input_ids = []
        input_elements = [e for e in self.model_elements if e.type == ElementTypes.INPUT]

        if len(input_elements) == 0:
            return None

        for el in input_elements:
            if model_instance is None:
                input_ids.append(self.get_pyomo_element_id(el.name))
            else:
                input_ids.append(self.get_pyomo_element_id(el.name))
        return input_ids
