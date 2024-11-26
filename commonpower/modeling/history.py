"""
Model Element History.
"""
from __future__ import annotations

import re
from copy import copy, deepcopy
from datetime import datetime
from typing import Dict, List, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.core import ConcreteModel, Param, Var

from commonpower.modeling.base import ElementTypes, ModelElement, ModelEntity


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

        # we need to shift the timestamps if t_index is not 0
        t_shift = t_index * (self.history[1][0] - self.history[0][0])

        filtered_history = []
        for t in self.history:
            filtered_history.append(
                (
                    t[0] + t_shift,
                    {key: val[t_index] if isinstance(val, np.ndarray) else val for key, val in t[1].items()},
                )
            )

        new_history = self.__class__(self.model_entities)
        new_history.history = filtered_history

        return new_history

    def to_df(self, at_time_index: int = 0) -> pd.DataFrame:
        """
        Converts the history to a pandas DataFrame.
        All available elements are columns, the timestamps are the index.

        Args:
            at_time_index (int, optional): Index of the forecast horizon to select for each time step.
                An index of 0 represents the realized values,
                a value of 1 represents the values that were predicted one time step into the future, etc.
                Defaults to 0.

        Returns:
            pd.DataFrame: Pandas DataFrame representation of the history.
        """
        # check if we already have single time index
        if isinstance(next(iter(self.history[0][1].values())), np.ndarray):
            new_history = self.filter_for_time_index(at_time_index)
        else:
            new_history = self

        return pd.DataFrame(
            data=[x[1] for x in new_history.history],
            index=[x[0] for x in new_history.history],
            columns=[x for x in new_history.history[0][1].keys()],
        )

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

        NON_STATE_ELEMENTS = [ElementTypes.VAR, ElementTypes.CONSTANT, ElementTypes.DATA, ElementTypes.INPUT]

        element_types = self._get_model_element_types()

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

                # for non-state variables we do not consider the terminal time step
                vals = vals[:-1] if element_types[element_id] in NON_STATE_ELEMENTS else vals

                time_series[label] = vals

                plot_args = {}
                for pat, style in plot_styles.items():
                    if re.search(pat, element_id):
                        plot_args = style
                        break

                m_type = element_types[element_id]
                default_style = 'stairs' if m_type in NON_STATE_ELEMENTS else ''

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

    def _get_model_element_types(self) -> dict[str, ElementTypes]:
        el_types = {}

        entities = self._get_entity_tree()
        for e in entities:
            me: ModelElement
            for me in e.model_elements:
                el_types[e.get_pyomo_element_id(me.name)] = me.type

        return el_types

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
        DEPRECATED! Use .filter_for_entities() and .filter_for_element_names() instead. \\
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
