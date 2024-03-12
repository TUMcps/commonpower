"""
Funcionality to generate new or augment existing networks in a partially randomized way.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TypeVar, Union

import numpy as np
from numpy.random import rand

from commonpower.core import Bus, Component, System
from commonpower.utils.param_initialization import ParamInitializer

T = TypeVar("T")


class Sampler:
    def __init__(self, sample_fcn: callable, **sample_fcn_kwargs):
        """
        Samplers provides a wrapper for a random sampling function (recommended: numpy.random).

        Args:
            sample_fcn (callable): Sampling function to use.
            **sample_fcn_kwargs: Keyword arguments to be passed to the sampling function.
        """
        self.sample_fcn = sample_fcn
        self.sample_fcn_kwargs = sample_fcn_kwargs

    def sample(self) -> Union[T, tuple[T]]:
        """
        Returns sampling result.

        Returns:
            Union[T, tuple[T]]: Sapled value(s).
        """
        sample = self.sample_fcn(**self.sample_fcn_kwargs)
        return tuple(sample.tolist()) if isinstance(sample, np.ndarray) else sample


class Factory:
    def __init__(self) -> Factory:
        """
        Factories allow generating a large number of busses with several sub-components.
        So-called meta configurations can be defined which specify how to sample
        bounds and parameters of each bus/component. Components are described by templates which
        contain their class, meta config, and the probability that a bus "owns" an instance.
        To attach multiple components of the same type to the same bus, multiple templates must be defined.
        """
        self.component_templates = []
        self.node_template = {}

    def generate_households(self, parent: Union[System, Bus], n_households: int):
        """
        Generates a number of "households" (busses) and attaches sub-components
        based on the available meta configurations.
        At the moment all the generated nodes are directly connected to a given parent without
        power lines being generated.

        Args:
            parent (Union[System, Bus]): Parent entity to attach the households to.
            n_households (int): Number of households to generate.
        """

        for i in range(n_households):
            n1 = self.node_template["type"](
                f"Household_{str(i)}", self._sample_config(self.node_template["meta_config"])
            )
            if self.node_template["data_providers"]:
                for ds in self.node_template["data_providers"]:
                    n1.add_data_provider(ds.sample() if hasattr(ds, "sample") else ds)

            parent.add_node(deepcopy(n1))

            for c in self.component_templates:
                done = False

                while not done:
                    if rand() > c["prob"]:  # component not selected
                        break

                    if c["meta_config"]:
                        config = self._sample_config(c["meta_config"])
                    else:
                        config = {}

                    c1 = c["type"](name=c["type"].__name__, config=config)

                    if c["data_providers"]:
                        for ds in c["data_providers"]:
                            c1.add_data_provider(ds.sample() if hasattr(ds, "sample") else ds)

                    # add component
                    parent.nodes[-1].add_node(deepcopy(c1))

                    done = True if c["multiple"] is False else False

    def fill_topology(self, sys: System) -> None:
        """
        Takes a system instance as argument and attaches components to all top-level nodes.

        Args:
            sys (System): System instance.
        """

        for node in sys.nodes:
            for c in self.component_templates:
                done = False

                while not done:
                    if rand() > c["prob"]:  # component not selected
                        break

                    if c["meta_config"]:
                        config = self._sample_config(c["meta_config"])
                    else:
                        config = {}

                    c1 = c["type"](name=c["type"].__name__, config=config)

                    if c["data_providers"]:
                        for ds in c["data_providers"]:
                            c1.add_data_provider(ds.sample() if hasattr(ds, "sample") else ds)

                    # add component
                    node.add_node(deepcopy(c1))

                    done = True if c["multiple"] is False else False

    def attach_node(self, parent: Union[System, Bus], node_constructor_kwargs: dict, node_config: dict) -> None:
        """
        Attaches a fully configured node to the given parent entity.

        Args:
            parent (Union[System, Bus]): Parent bus/sys.
            node_constructor_kwargs (dict): Constructor arguments for the generated node.
            node_config (dict): Config dict for the generated node.
        """

        sampled_node_config = self._sample_config(self.node_template["meta_config"])
        n1 = self.node_template["type"](**node_constructor_kwargs, config={**sampled_node_config, **node_config})
        if self.node_template["data_providers"]:
            for ds in self.node_template["data_providers"]:
                n1.add_data_provider(ds.sample() if hasattr(ds, "sample") else ds)

        parent.add_node(deepcopy(n1))

        for c in self.component_templates:
            done = False

            while not done:
                if rand() > c["prob"]:  # component not selected
                    break

                if c["meta_config"]:
                    config = self._sample_config(c["meta_config"])
                else:
                    config = {}

                c1 = c["type"](name=c["type"].__name__, config=config)

                if c["data_providers"]:
                    for ds in c["data_providers"]:
                        c1.add_data_provider(ds.sample() if hasattr(ds, "sample") else ds)

                # add component
                parent.nodes[-1].add_node(deepcopy(c1))

                done = True if c["multiple"] is False else False

    def set_bus_template(
        self,
        bus_type: Bus,
        meta_config: Union[
            None, dict[str, Union[Sampler, list[ParamInitializer, dict[str, Union[Sampler, str]]]]]
        ] = None,
        data_providers: Union[None, list[Sampler]] = None,
    ) -> None:
        """
        The meta-config defines how to sample the config values for all node instances.
        The entries must have one of the following forms:
        - Sampler for scalar constants and ranges with appropriate Sampler configuration
        - [ParamInitializer class, {"attr": Sampler}] for param initializers.
            Everything in the dict will be sampled and passed as **kwargs to the created ParamInitializer instance.

        Args:
            bus_type (Bus): Class of the bus.
            meta_config (Union[None, dict[str, Union[Sampler, list[ParamInitializer,
                dict[str, Union[Sampler, str]]]]]], optional): Meta configuration for the node.
                For examples, please refer to the Tutorials. Defaults to None.
            data_providers (Union[None, list[Sampler]], optional): List of Samplers each sampling
                from a selection of data sources. The sampled data sources are attached to the node.
                Defaults to None.
        """
        if meta_config:
            for param, source in meta_config.items():
                if isinstance(source, list):
                    assert issubclass(source[0], ParamInitializer) or (
                        isinstance(source[0], (int, float)) and isinstance(source[1], (int, float))
                    ), f"We expect a different config for {param}, not {source[0].__class__.__name__}"
                else:
                    assert isinstance(source, (int, float, Sampler)), (
                        f"Source for config param {param} must be of type int/float/{Sampler.__name__} instead of"
                        f" {source.__class__.__name__}"
                    )

        self.node_template = {"type": bus_type, "meta_config": meta_config, "data_providers": data_providers}

    def add_component_template(
        self,
        component_type: Component,
        probability: float,
        multiple_allowed=False,
        meta_config: Union[
            None, dict[str, Union[Sampler, list[ParamInitializer, dict[str, Union[Sampler, str]]]]]
        ] = None,
        data_providers: Union[None, list[Sampler]] = None,
    ) -> None:
        """
        The meta-config defines how to sample the config values for component instances.
        The entries must have one of the following forms:
        - Sampler for scalar constants and ranges with appropriate Sampler configuration
        - [ParamInitializer class, {"attr": Sampler}] for param initializers.
            Everything in the dict will be sampled and passed as **kwargs to the created ParamInitializer instance.

        Args:
            component_type (Component): Class of the component.
            probability (float): Probability that an instance of this component is attached to a household.
            multiple_allowed (bool, optional): If True, multiple components can be added to a node.
                This means we do the random sampling repeatedly based on the given probability threshold.
            meta_config (Union[None, dict[str, Union[Sampler, list[ParamInitializer,
                dict[str, Union[Sampler, str]]]]]], optional): Meta configuration for the component.
                For examples, please refer to the Tutorials. Defaults to None.
            data_providers (Union[None, list[Sampler]], optional): List of Samplers each sampling
                from a selection of data sources. The sampled data sources are attached to the component.
                Defaults to None.
        """

        assert abs(probability) <= 1.0, "Probability must be in [0,1]"
        if meta_config:
            for param, source in meta_config.items():
                if isinstance(source, (tuple, list)):
                    assert (isinstance(source[0], (int, float)) and isinstance(source[1], (int, float))) or (
                        issubclass(source[0], ParamInitializer)
                    ), f"We expect a different config for {param}, not {source[0].__class__.__name__}"
                else:
                    assert isinstance(source, (int, float, Sampler)), (
                        f"Source for config param {param} must be of type int/float/{Sampler.__name__} instead of"
                        f" {source.__class__.__name__}"
                    )

        self.component_templates.append(
            {
                "type": component_type,
                "prob": abs(probability),
                "multiple": multiple_allowed,
                "meta_config": meta_config,
                "data_providers": data_providers,
            }
        )

    def _sample_config(self, meta_config: dict) -> dict:
        """
        This method samples a config from a given meta config

        Args:
            meta_config (dict): Meta config to use.

        Returns:
            dict: Sampled config.
        """
        config = {}
        for param, source in meta_config.items():
            if isinstance(source, (tuple, list)):
                if isinstance(source[0], (int, float)) and isinstance(source[1], (int, float)):  # e.g. var bounds
                    config[param] = source
                elif issubclass(source[0], ParamInitializer):  # this is a ParamInitializer config
                    initializer_class = source[0]
                    initializer_args = source[1]
                    initializer_kwargs = {}
                    for attr, arg in initializer_args.items():
                        if isinstance(arg, Sampler):
                            initializer_kwargs[attr] = arg.sample()
                        else:
                            initializer_kwargs[attr] = arg
                    config[param] = initializer_class(**initializer_kwargs)
                else:
                    raise NotImplementedError(f"We did not expect {source} as config for {param}")
            elif isinstance(source, Sampler):
                config[param] = source.sample()
            elif isinstance(source, (int, float)):
                config[param] = source
            else:
                raise NotImplementedError(f"We did not expect {source} as config for {param}")

        return config
