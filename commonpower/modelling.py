"""
Generic abstractions and functionality for interacting with the pyomo layer.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import OrderedDict
from copy import copy, deepcopy
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Type, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from prettytable import PrettyTable
from pyomo.core import Block, ConcreteModel, Constraint, Expression, Objective, Param, Set, Var

from commonpower.utils import model_root, rgetattr, rhasattr, rsetattr
from commonpower.utils.cp_exceptions import EntityError
from commonpower.utils.param_initialization import ParamInitializer


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
    #: Cost variable. This is essentially a generic variable but explicitly defined to simplify
    #: downstream analysis. Maps to the Pyomo type Var.
    COST = 7
    #: Set. Sets can be useful to specify the values a discrete variable can take. Maps to the Pyomo type Set.
    SET = 8


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
            self.pyomo_class = Param
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
        elif self.type == ElementTypes.CONSTRAINT:
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


class ModelHistory:
    def __init__(self, model_entities: list[ModelEntity], retention: int = -1) -> ModelHistory:
        """
        This class provides a lightweight interface to log "snapshots" of a pyomo model and
        some methods to retrieve this information.
        The logs are stored in self.history in the form:
            [(<time stamp>, {<global model element id>: <value>, ...}), ...].

        Args:
            model_entities (list[ModelEntity]): Model entities to create a history for.
                Note that all Vars/Params of the entity and all its subordinate entities will be included.
                Technically, we are including everything within the pyomo blocks that correspond to the given entities.
            retention (int, optional): How many logs are kept before deleting from the top (essentially a ring buffer).
                When set to -1, all logs will be kept. Defaults to -1.
        """
        self.model_entities = copy(model_entities)
        self.retention = retention

        self.history = []

    def log(self, model: ConcreteModel, timestamp: Union[datetime, str, int]) -> None:
        """
        Creates a "snapshot" of the values of all model elements corresponding to the given entities and
        stores them together with the given timestamp.
        If self.history is already "full" (specified by self.retention), the first entry of self.history is deleted.

        Args:
            model (ConcreteModel): Model to extract the values from.
            timestamp (Union[datetime, str, int]): Timestamp information.
                Can technically be of any type but should be unique to avoid downstream issues.
        """
        snapshot = {}

        for ent in self.model_entities:
            local_model = ent.get_self_as_pyomo_block(model)
            for el in local_model.component_objects([Var, Param], active=True):
                snapshot[el.name] = ent.get_value(local_model, el.name)

        if self.retention > 0 and len(self.history) >= self.retention:
            self.history.pop(0)

        self.history.append((timestamp, deepcopy(snapshot)))

    def reset(self) -> None:
        """
        Clears self.history.
        """
        self.history = []

    def filter_for_entities(
        self, entities: Union[ModelEntity, List[ModelEntity]], follow_node_tree: bool = False
    ) -> ModelHistory:
        """
        Filters the history to only contain data from the given entity instances.

        Args:
            entities (Union[ModelEntity, List[ModelEntity]]): Entites to filter for.
            follow_node_tree (bool, optional): If True, all entites which are subordinate
                to the given entites will be included. Defaults to False.

        Returns:
            ModelHistory: Filtered model history.
        """
        if not isinstance(entities, list):
            entities = [entities]

        if follow_node_tree is True:
            entities = self._get_entity_tree(entities)

        filtered_history = self._filter_history_for_entities(entities)

        new_history = self.__class__(entities)
        new_history.history = filtered_history

        return new_history

    def filter_for_entity_types(self, entity_types: Union[Type[ModelEntity], List[Type[ModelEntity]]]) -> ModelHistory:
        """
        Filters the history to only contain entities of the given types.

        Args:
            entity_types (Union[Type[ModelEntity], List[Type[ModelEntity]]]): Entity types to filter for.

        Returns:
            ModelHistory: Filtered model history.
        """
        if not isinstance(entity_types, list):
            entity_types = [entity_types]

        entities = self._get_entity_tree()

        filtered_entities = [e for e in entities if any([isinstance(e, t) for t in entity_types])]

        filtered_history = self._filter_history_for_entities(filtered_entities)

        new_history = self.__class__(filtered_entities)
        new_history.history = filtered_history

        return new_history

    def filter_for_element_names(self, names: Union[str, List[str]]) -> ModelHistory:
        """
        Filters the history to only contain model elements of the given local names.

        Args:
            names (Union[str, List[str]]): Local names to filter for.

        Returns:
            ModelHistory: Filtered model history.
        """
        if not isinstance(names, list):
            names = [names]

        entities = self._get_entity_tree()

        filtered_history = []
        for t in self.history:
            filtered_history.append(
                (
                    t[0],
                    {
                        key: val
                        for key, val in t[1].items()
                        if any([e.get_pyomo_element_id(name) == key for e in entities for name in names])
                    },
                )
            )

        new_history = self.__class__(self.model_entities)
        new_history.history = filtered_history

        return new_history

    def filter_for_time_index(self, t_index: int = 0) -> ModelHistory:
        """
        Filters all element histories for a certain time index.

        Args:
            t_index (int, optional): Time index. A time index of 0 represents the realized values at each timestep.
                Defaults to 0.

        Returns:
            ModelHistory: Filtered model history.
        """
        filtered_history = []
        for t in self.history:
            filtered_history.append(
                (t[0], {key: val[t_index] if isinstance(val, np.ndarray) else val for key, val in t[1].items()})
            )

        new_history = self.__class__(self.model_entities)
        new_history.history = filtered_history

        return new_history

    def filter_for_time_period(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> ModelHistory:
        """
        Filters all element histories for a given time period

        Args:
            start (Union[str, pd.Timestamp]): beginning of the time period.
            If str, should be in format "2016-09-04 00:00:00".
            end (Union[str, pd.Timestamp]): end of the time period. If str, should be in format "2016-09-04 00:00:00".

        Returns:
            (ModelHistory): the filtered history.

        """
        filtered_history = []
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        time_stamps = [t[0] for t in self.history]
        start_index = [i for i in range(len(time_stamps)) if time_stamps[i] == start]
        end_index = [i for i in range(len(time_stamps)) if time_stamps[i] == end]
        for t in range(end_index[0] - start_index[0] + 1):
            filtered_history.append(self.history[start_index[0] + t])

        new_history = self.__class__(self.model_entities)
        new_history.history = filtered_history

        return new_history

    def plot(
        self,
        histories: Union[ModelHistory, List[ModelHistory]] = [],
        timestamp_format: str = "%Y-%m-%d %H:%M",
        return_time_series=False,
        show: bool = True,
        x_label_interval=1,
        plot_styles: Dict[str, dict] = {},
        **plt_show_kwargs,
    ) -> Union[None, dict]:
        """
        Plots entire history and, if given, even multiple histories.
        We assume here that all elements have been consistently logged within the histories.

        Args:
            histories (Union[ModelHistory, List[ModelHistory]], optional): Additional histories to plot.
                Defaults to [].
            timestamp_format (str, optional): Format to display the timestamp in. Defaults to "%Y-%m-%d %H:%M".
            return_time_series (bool, optional): If true, returns the time series of realized element values.
                Defaults to False.
            show (bool, optional): Determines if the plot is shown. Defaults to True.
            x_label_interval (int, optional): Only print labels on the x-axis every n timesteps (to reduce clutter)
            plot_styles (dict[str, dict], optional): Dictionary of regular expressions to `pyplot.plot` kwargs.
                For every element that is plotted, the id is matched (re.search) against all keys of this dict.
                The kwargs of the first match are used for the call to plot.
                Additionally, a `drawstyle` of `stairs` is supported, which calls `pyplot.stairs` instead of `plot`.
                (VAR, CONSTANT, DATA and INPUT default to `stairs`)
                Example:
                ```
                history.plot(plot_styles={
                    'soc': {  # color all elements that have "soc" in their id green
                        'color': 'green',
                    },
                    'p$': {  # draw all elements that end in "...p" as dotted lines
                        'linestyle': ':',
                        'alpha': 0.5,
                    },
                    '': {  # fallback: draw all remaining as lines even if they would default to "stairs"
                        'drawstyle': 'default',
                    },
                })
                ```
        """

        time_series = {}

        if not isinstance(histories, list):
            histories = [histories]

        legend_labels = []
        for idx, hist in enumerate([self] + histories):
            for element_id in hist.history[0][1].keys():
                label = f"Hist {idx}: {element_id}" if len(histories) > 1 else element_id
                legend_labels.append(label)
                vals = [
                    t[1][element_id][0] if isinstance(t[1][element_id], np.ndarray) else t[1][element_id]
                    for t in hist.history
                ]  # only realized values
                time_series[label] = vals

                plot_args = {}
                for pat, style in plot_styles.items():
                    if re.search(pat, element_id):
                        plot_args = style
                        break

                m_type = self._get_model_element_type(element_id)
                default_style = (
                    'stairs'
                    if m_type in [ElementTypes.VAR, ElementTypes.CONSTANT, ElementTypes.DATA, ElementTypes.INPUT]
                    else ''
                )

                if plot_args.get('drawstyle', default_style) == 'stairs':
                    plot_args.pop('drawstyle', None)
                    plt.stairs(vals, range(len(vals) + 1), baseline=None, **plot_args)
                else:
                    plt.plot(range(len(vals)), vals, **plot_args)

        x_labels_full = [x[0].strftime(timestamp_format) if isinstance(x[0], datetime) else x[0] for x in self.history]
        x_labels = [''] * len(self.history)
        x_labels[::x_label_interval] = x_labels_full[::x_label_interval]
        plt.xticks(
            ticks=range(len(self.history)),
            labels=x_labels,
        )
        plt.xticks(rotation=90, ha="center")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend(legend_labels)
        plt.title("Element Realization")
        plt.tight_layout()

        if show is True:
            plt.show(**plt_show_kwargs)

        if return_time_series is True:
            return time_series

    def _get_entity_tree(self, entities: list[ModelEntity] = None) -> list[ModelEntity]:
        entities = copy(self.model_entities) if entities is None else copy(entities)
        tmp = []
        for ent in entities:
            tmp += ent.get_children()
        entities += tmp

        return entities

    def _get_model_element_type(self, id: str) -> ElementTypes | None:
        entities = self._get_entity_tree()

        el_name = id.split(".")[-1]

        for e in entities:
            if e.get_pyomo_element_id(el_name) == id:
                me: ModelElement
                for me in e.model_elements:
                    if me.name == el_name:
                        return me.type

        return None

    def _filter_history_for_entities(self, entities: list[ModelEntity]) -> list[tuple]:
        filtered_history = []
        for t in self.history:
            filtered_history.append(
                (
                    t[0],
                    {
                        key: val
                        for key, val in t[1].items()
                        if any([e.get_pyomo_element_id(key.split(".")[-1]) == key for e in entities])
                    },
                )
            )
        return filtered_history

    def __repr__(self) -> str:
        """
        Returns self.history as string.

        Returns:
            str: str(self.history)
        """
        return str(self.history)

    def get_history_for_element(
        self, entity: ModelEntity, name: str, only_realized_values=True
    ) -> list[tuple[str, Union[int, float, np.ndarray]]]:
        """
        DEPRECIATED! Use .filter_for_entities() and .filter_for_element_names() instead. \\
        Interface to extract the history of a single model element.

        Args:
            entity (ModelEntity): Entity the element is associated with.
            name (str): Local name of the element. This is a utility since the elements are stored
                in the history with their global id.
            only_realized_values (bool, optional): Every log of an indexed element is a np.ndarray.
                If this argument is set to True, only the first element of this array is retrieved for every log.
                The intuition is that in an MPC-type setup, only the value at time index 0 is
                actually realized (the rest only "predicted"). Defaults to True.

        Returns:
            list[tuple[str, Union[int, float, np.ndarray]]]:
                Element history in the form: [(<time stamp>, <values(s)>, ...].
        """
        history = []
        element_id = entity.get_pyomo_element_id(name)
        for t in self.history:
            val = (
                t[1][element_id][0]
                if isinstance(t[1][element_id], np.ndarray) and only_realized_values is True
                else t[1][element_id]
            )
            history.append((t[0], val))

        return history

    def plot_realization(
        self,
        entities: Union[ModelEntity, List[ModelEntity]],
        names: Union[str, List[str]],
        follow_node_tree: bool = False,
        **plt_show_kwargs,
    ) -> None:
        """
        DEPRECIATED! Use .plot() instead. \\
        Lightweight interface to plot the realized history of a single or multiple model element(s).
        The output is a pyplot line plot.

        Args:
            entities (ModelEntity): Entities the elements are associated with.
            names (str): Local names of the elements. This is a utility since the elements are
                stored in the history with their global id.
            follow_node_tree (bool): If True, every matching model element in the node tree below
                the given entities is plotted. Defaults to False.
        """

        if isinstance(names, str):
            entities = [entities]
            names = [names]

        if follow_node_tree is True:
            # fetch all nodes in the node tree
            tmp = []
            for ent in entities:
                tmp += ent.get_children()
            entities += tmp
            # expand name list (we only use the first name provided and ignore anything else)
            names = np.repeat(names[0], len(entities))

        valid_entitites = []
        for i in range(len(entities)):
            try:
                vals = self.get_history_for_element(entities[i], names[i])
                valid_entitites.append(entities[i])
                plt.plot(range(len(vals)), [x[1] for x in vals])
            except KeyError:
                # this entity does not have an element of that name
                pass

        plt.xticks(ticks=range(len(vals)), labels=[x[0] for x in vals])
        plt.xticks(rotation=45)
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend(
            [
                f"{valid_entitites[i].get_pyomo_element_id(names[i])} ({valid_entitites[i].name})"
                for i in range(len(valid_entitites))
            ]
        )
        # plt.title(f"{entity.get_pyomo_element_id(name)} ({entity.name})")
        plt.title("Element Realization")
        plt.tight_layout()
        plt.show(**plt_show_kwargs)


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
            if (
                el.type == ElementTypes.CONSTANT and el.initialize is None
            ):  # "constants" can be defined either by a constant float or a ParamInitializer which is called on reset()
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

        self.model_elements = []
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

        root_model = model_root(model)  # get root model
        local_id = name.split(".")[-1] if self.id else name  # get local element id (do nothing if self is system)
        global_id = self.get_pyomo_element_id(local_id)  # get gobal element id

        # Check if name is on the branch of self.
        # This makes sure that we catch misuse, e.g., name 'n13.p' if self is 'n12'
        if name not in global_id:
            raise EntityError(self, f"The variable {name} is not on the model branch of the calling entity")

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

        raise EntityError(self, f"The variable {global_id} could not be found in the given model")

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

    def cost_fcn(self, model: ConcreteModel, t: int = 0) -> Expression:
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

    def observation_space(self, obs_mask: dict):
        """
        Determines the observation space of an entity based on the observation mask by retrieving
        the bounds of the model elements listed in the mask

        Args:
            obs_mask (dict): dictionary containing the IDs of model elements which should be observed

        Returns:
            None/gym.spaces.Dict: None if the node has no elements that should be observed, else a dictionary as in
            {model element ID: box observation space}

        """
        # ToDo: check type of variables/data --> if they are binary, we cannot use box spaces?
        # for now all model elements with type DATA and STATE are observations
        obs = [e for e in self.model_elements if e.name in obs_mask[self.id]]
        lower = {}
        upper = {}
        for e in obs:
            pyomo_el = self.get_pyomo_element(e.name, self.instance)
            # for states, we only want to observe the first element
            if e.type == ElementTypes.STATE:
                if e.bounds is not None:
                    lower[e.name] = e.bounds[0]
                    upper[e.name] = e.bounds[1]
                else:
                    lower[e.name] = -np.inf
                    upper[e.name] = np.inf
            else:
                if e.bounds is not None:
                    lower[e.name] = (
                        [e.bounds[0] for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else e.bounds[0]
                    )
                    upper[e.name] = (
                        [e.bounds[1] for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else e.bounds[1]
                    )
                else:
                    lower[e.name] = [-np.inf for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else -np.inf
                    upper[e.name] = [np.inf for idx in pyomo_el.index_set()] if pyomo_el.is_indexed() else np.inf

        if lower:
            obs_space = gym.spaces.Dict(
                {
                    el.name: gym.spaces.Box(
                        low=np.array([lower[el.name]]).reshape((-1,)),
                        high=np.array([upper[el.name]]).reshape((-1,)),
                        dtype=np.float64,
                    )
                    for el in obs
                }
            )

            return obs_space
        else:
            return None

    def observe(self, obs_mask: Dict) -> dict:
        """
        Get observations for one node within the system based on the model items within the observation mask.

        Args:
            obs_mask (dict): dictionary containing the IDs of model elements which should be observed

        Returns:
            dict: dict of observed values as {element ID: value}

        """
        obs = OrderedDict()
        for el in self.model_elements:
            if el.name in obs_mask[self.id]:
                # for states, we only want to get the current value
                if el.type == ElementTypes.STATE:
                    obs[el.name] = np.array(self.get_value(self.instance, el.name))[0].reshape((1,))
                else:
                    obs[el.name] = np.array(self.get_value(self.instance, el.name))

        if len(obs) == 0:
            return None
        else:
            return obs

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


class MIPExpressionBuilder:
    def __init__(self, entity: ModelEntity, M: int = 1e3, eps: float = 1e-5):
        """
        The expression builder allows to convert logical expression into mixed integer constraints.
        In the process it also creates all necessary auxiliary variables.
        The structure of of interface is as follows:
            - Create expression builder instance.
            - Generate expressions. Constraints and an output variable are created
              (or an existing output variable referenced) for each expression.
              The corresponding ModelElements are internally stored in self.model_elements.
            - Obtain all generated ModelElements from self.model_elements.

        The MIP conversions here are based on

        @article{brown2007formulating,
            title={Formulating integer linear programs: A rogues' gallery},
            author={Brown, Gerald G and Dell, Robert F},
            journal={INFORMS Transactions on Education},
            volume={7},
            number={2},
            pages={153--159},
            year={2007},
            publisher={INFORMS}
        }

        IMPORTANT NOTE: The Integrality Tolerance of the used solver has to be set such that
        IntFeasTol * M < eps for all the constraints to work correctly.

        Args:
            entity (ModelEntity): Entity used to obtain referenced pyomo model elements from.
            M (int, optional): Constant for bigM constraints. Defaults to 1e3.
            eps (float, optional): Slack value for strict inequalities (pyomo only allows for <=, >=, ==).
                Defaults to 1e-5.
        """
        self.vars = []
        self.model_elements = []
        self.entity = entity
        self.M = M
        self.eps = eps  # this is a slack value because pyomo only allows for <=, >=, ==

    def from_geq(
        self, a: Union[str, int, float], b: Union[str, int, float], out: str = None, is_new: bool = True, M: int = None
    ) -> str:
        """
        Generates constraints based on: \\
        out = 1 if a >= b, out = 0 otherwise.
        The MILP formulation using bigM constraints is: \\
        a >= b - M*(1-out) \\
        a < b + M*out (we use: a + eps <= b + M*out)

        Args:
            a (Union[str, int, float]): Left hand side of the inequality.
            b (Union[str, int, float]): Right hand side of the inequality.
            out (str, optional): Name of the output variable.
                If not given, a name is autogenerated ('aux_' + 5 hex characters).
            is_new (bool, optional): Indicates if the variable is new. If so, a corresponding ModelElement added.
                Defaults to True.
            M (int, optional): Constant M for bigM type constraints. If not given, the class' M is used.

        Returns:
            str: Name of the output variable
        """

        if not out:
            is_new = True  # just in case someone tried to be nasty...
            out = "aux_" + "%05x" % random.randrange(16**5)  # 5 hex characters

        if is_new:
            self.model_elements.append(
                ModelElement(out, ElementTypes.VAR, "auxiliary binary variable", domain=pyo.Binary)
            )

        if not M:
            M = self.M

        def cbin1_f(model, t):
            return self._parse_var(a, model, t) >= self._parse_var(b, model, t) - M * (
                1 - self._parse_var(out, model, t)
            )

        def cbin2_f(model, t):
            return self._parse_var(a, model, t) + self.eps <= self._parse_var(b, model, t) + M * self._parse_var(
                out, model, t
            )

        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_1", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin1_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_2", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin2_f
            )
        )

        return out

    def from_gt(
        self, a: Union[str, int, float], b: Union[str, int, float], out: str = None, is_new: bool = True, M: int = None
    ) -> str:
        """
        Generates constraints based on: \\
        out = 1 if a > b, out = 0 otherwise.
        The MILP formulation using bigM constraints is: \\
        a - eps >= b - M*(1-out) \\
        a <= b + M*out

        Args:
            a (Union[str, int, float]): Left hand side of the inequality.
            b (Union[str, int, float]): Right hand side of the inequality.
            out (str, optional): Name of the output variable.
                If not given, a name is autogenerated.
            is_new (bool, optional): Indicates if the variable is new. If so, a corresponding ModelElement added.
                Defaults to True.
            M (int, optional): Constant M for bigM type constraints. If not given, the class' M is used.

        Returns:
            str: Name of the output variable
        """

        if not out:
            is_new = True  # just in case...
            out = "aux_" + "%05x" % random.randrange(16**5)

        if is_new:
            self.model_elements.append(
                ModelElement(out, ElementTypes.VAR, "auxiliary binary variable", domain=pyo.Binary)
            )

        if not M:
            M = self.M

        def cbin1_f(model, t):
            return self._parse_var(a, model, t) - self.eps >= self._parse_var(b, model, t) - M * (
                1 - self._parse_var(out, model, t)
            )

        def cbin2_f(model, t):
            return self._parse_var(a, model, t) <= self._parse_var(b, model, t) + M * self._parse_var(out, model, t)

        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_1", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin1_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_2", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin2_f
            )
        )

        return out

    def from_and(self, a: str, b: str, out: str = None, is_new: bool = True) -> str:
        """
        Generates constraints based on: \\
        out = 1 if (a and b), out = 0 otherwise.
        The MILP formulation is: \\
        out >= a + b - 1 \\
        out <= a \\
        out <= b

        Args:
            a (str): Variable 1 (must be binary).
            b (str): Variable 2 (must be binary).
            out (str, optional): Name of the output variable.
                If not given, a name is autogenerated.
            is_new (bool, optional): Indicates if the variable is new. If so, a corresponding ModelElement added.
                Defaults to True.
            If not given, a name is autogenerated and a corresponding ModelElement added.

        Returns:
            str: Name of the output variable.
        """

        if not out:
            is_new = True  # just in case...
            out = "aux_" + "%05x" % random.randrange(16**5)

        if is_new:
            self.model_elements.append(
                ModelElement(out, ElementTypes.VAR, "auxiliary binary variable", domain=pyo.Binary)
            )

        def cbin1_f(model, t):
            return self._parse_var(out, model, t) >= self._parse_var(a, model, t) + self._parse_var(b, model, t) - 1

        def cbin2_f(model, t):
            return self._parse_var(out, model, t) <= self._parse_var(a, model, t)

        def cbin3_f(model, t):
            return self._parse_var(out, model, t) <= self._parse_var(b, model, t)

        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_1", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin1_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_2", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin2_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_3", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin3_f
            )
        )

        return out

    def from_or(self, a: str, b: str, out: str = None, is_new: bool = True) -> str:
        """
        Generates constraints based on: \\
        out = 1 if (a or b), out = 0 otherwise.
        The MILP formulation is: \\
        out <= a + b \\
        out >= a \\
        out >= b

        Args:
            a (str): Variable 1 (must be binary).
            b (str): Variable 2 (must be binary).
            out (str, optional): Name of the output variable.
                If not given, a name is autogenerated.
            is_new (bool, optional): Indicates if the variable is new. If so, a corresponding ModelElement added.
                Defaults to True.
            If not given, a name is autogenerated and a corresponding ModelElement added.

        Returns:
            str: Name of the output variable.
        """

        if not out:
            is_new = True  # just in case...
            out = "aux_" + "%05x" % random.randrange(16**5)

        if is_new:
            self.model_elements.append(
                ModelElement(out, ElementTypes.VAR, "auxiliary binary variable", domain=pyo.Binary)
            )

        def cbin1_f(model, t):
            return self._parse_var(out, model, t) <= self._parse_var(a, model, t) + self._parse_var(b, model, t)

        def cbin2_f(model, t):
            return self._parse_var(out, model, t) >= self._parse_var(a, model, t)

        def cbin3_f(model, t):
            return self._parse_var(out, model, t) >= self._parse_var(b, model, t)

        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_1", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin1_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_2", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin2_f
            )
        )
        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_3", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin3_f
            )
        )

        return out

    def from_not(self, a: str, out: str = None, is_new: bool = True) -> str:
        """
        Generates a constraint based on: \\
        out = 1 - a
        The MILP formulation is: \\
        out == 1 - a

        Args:
            a (str): Variable 1 (must be binary)
            out (str, optional): Name of the output variable.
                If not given, a name is autogenerated.
            is_new (bool, optional): Indicates if the variable is new.
                If so, a corresponding ModelElement added. Defaults to True.
            If not given, a name is autogenerated and a corresponding ModelElement added.

        Returns:
            str: Name of the output variable.
        """

        if not out:
            is_new = True  # just in case...
            out = "aux_" + "%05x" % random.randrange(16**5)

        if is_new:
            self.model_elements.append(
                ModelElement(out, ElementTypes.VAR, "auxiliary binary variable", domain=pyo.Binary)
            )

        def cbin1_f(model, t):
            return self._parse_var(out, model, t) == 1 - self._parse_var(a, model, t)

        self.model_elements.append(
            ModelElement(
                f"cmilp_{out}_1", ElementTypes.CONSTRAINT, f"constrain auxiliary binary variable {out}", expr=cbin1_f
            )
        )

        return out

    def enforce_value(self, a: str, val: Union[int, float]) -> None:
        """
        Generates an always-true constraint for a: \\
        val == a

        Args:
            a (str): Variable 1.
            val (Union[int, float]): value the variable should be set to.
        """

        def cbin1_f(model, t):
            return self._parse_var(a, model, t) == val

        self.model_elements.append(
            ModelElement(f"cenf_{a}", ElementTypes.CONSTRAINT, f"constrain binary variable {a} to {val}", expr=cbin1_f)
        )

    def _parse_var(
        self, var: Union[str, int, float, list[int], list[float]], model: ConcreteModel, t: int
    ) -> Union[Union[Var, Param], int, float]:
        """
        Helper method to access elements from self.entity's pyomo model.

        Args:
            var (Union[str, int, float, list[int], list[float]]): Either variable name or constant value.
            model (ConcreteModel): Pyomo model to access.
            t (int): Time index.

        Returns:
            Union[Union[Var, Param], int, float]: Either pyomo element or constant value.
        """
        if isinstance(var, str):
            el = self.entity.get_pyomo_element(var, model)
            if el.is_indexed():
                return el[t]
            else:
                return el
        else:
            if isinstance(var, (list, tuple)):
                return var[t]
            else:
                return var
