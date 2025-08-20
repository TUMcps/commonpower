import hashlib
import io
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import Env
from pytupli.benchmark import TupliEnvWrapper
from pytupli.schema import ArtifactMetadata, RLTuple
from pytupli.storage import TupliStorage

from commonpower.control.wrappers import recursive_items
from commonpower.core import System
from commonpower.data_forecasting.data_sources import CSVDataSource, PandasDataSource


def get_all_data_providers(env: Env, node=None, provider=None) -> OrderedDict:
    """
    Retrieves all data providers from the environment.

    Args:
        env (Env): The environment instance

    Returns:
        list[str]: A list of data provider names
    """
    providers = OrderedDict()
    sys_inst = env.get_wrapper_attr('sys')
    node = sys_inst if node is None else node
    provider = OrderedDict() if provider is None else provider
    if hasattr(node, "data_providers"):
        provider.update({node.id + '.' + '.'.join(prov.observable_features): prov for prov in node.data_providers})
    if len(node.get_children()) > 0:
        for child in node.get_children():
            providers.update(get_all_data_providers(env, child, provider))
    else:
        providers.update(provider)
    return providers  # Use set to avoid duplicates


def remove_selected_data_sources(env: Env, to_remove: dict, node=None) -> Env:
    sys_inst = env.get_wrapper_attr('sys')
    node = sys_inst if node is None else node
    if hasattr(node, "data_providers"):
        providers = getattr(node, 'data_providers', {})
        for i, provider in enumerate(providers):
            data_source = getattr(provider, 'data', None)
            if isinstance(data_source, (PandasDataSource, CSVDataSource)):
                # remove this datasource and replace with hash
                data_source.data = to_remove[data_source]
    if len(node.get_children()) > 0:
        for child in node.get_children():
            remove_selected_data_sources(env, to_remove, child)


class CommonPowerRLTuple(RLTuple):
    @classmethod
    def from_env_step(
        cls,
        obs: Any,
        action: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> RLTuple:
        """
        Creates an RLTuple from environment step outputs.
        This method can be overridden to handle different types of observations and actions.

        Args:
            obs: The observation from the environment
            action: The action taken
            reward: The reward received
            terminated: Whether the episode terminated
            truncated: Whether the episode was truncated
            info: Additional information from the environment step

        Returns:
            RLTuple: A new tuple instance
        """
        # Convert dictionaries to numpy arrays if necessary
        if isinstance(obs, dict):
            all_obs = []
            for ctrl_obs in obs.values():
                new_obs = np.array([])
                for el_id, el_obs in recursive_items(ctrl_obs):
                    new_obs = np.concatenate((new_obs, el_obs))
                all_obs.append(new_obs)
            obs = np.array(all_obs).flatten()

        if isinstance(action, dict):
            all_actions = []
            for act in action.values():
                ctrl_action = []
                for node_action in act.values():
                    for el_action in node_action.values():
                        ctrl_action.append(el_action)
                all_actions.append(ctrl_action)
            action = np.array(all_actions).flatten()

        if isinstance(reward, dict):
            all_rewards = []
            for rew in reward.values():
                all_rewards.append(np.array(rew))
            reward = np.array(all_rewards)
        elif isinstance(reward, (int, float)):
            reward = np.array([reward])
        # Convert numpy arrays to lists for serialization
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        else:
            raise ValueError(f'Unsupported observation type: {type(obs)}. Expected numpy array.')
        if isinstance(action, (np.ndarray, np.integer)):
            action = action.tolist()
        else:
            raise ValueError(f'Unsupported action type: {type(action)}. Expected numpy array or int.')
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        else:
            raise ValueError(f'Unsupported reward type: {type(reward)}. Expected numpy array.')

        return cls(
            state=obs,
            action=action,
            reward=reward,
            terminal=terminated,
            timeout=truncated,
            info=info,
        )


class CommonPowerTupliEnvWrapper(TupliEnvWrapper):
    def _serialize(self, env):
        """
        Serialize environment with deterministic handling of data sources.
        """
        related_data_sources = []
        data_providers = get_all_data_providers(env)
        to_remove = dict()

        # Create deterministic mapping of data source content to avoid order dependency
        data_source_map = {}

        for prov_name, provider in data_providers.items():
            ds = getattr(provider, 'data', None)
            # We only have to prepare the Pandas and CSV data sources
            if isinstance(ds, (PandasDataSource, CSVDataSource)):
                df = getattr(ds, 'data', None)
                try:
                    # Create deterministic CSV content with sorted columns and consistent formatting
                    content = df.to_csv(encoding='utf-8', index=True, float_format='%.6f')
                    content_bytes = content.encode(encoding='utf-8')

                    # Create content hash for deduplication
                    content_hash = hashlib.sha256(content_bytes).hexdigest()

                    if content_hash not in data_source_map:
                        # First time seeing this content, store it
                        metadata = ArtifactMetadata(name=f"data_source_{content_hash[:8]}")
                        df_storage_metadata = self.storage.store_artifact(artifact=content_bytes, metadata=metadata)
                        data_source_map[content_hash] = df_storage_metadata.id
                        related_data_sources.append(df_storage_metadata.id)

                    # Replace data source with the deterministic artifact ID
                    to_remove[ds] = data_source_map[content_hash]

                except Exception as e:
                    raise ValueError(f'Failed to serialize data source {getattr(ds, "name", "unknown")}: {e}')

        # Apply the data source replacements
        remove_selected_data_sources(env, to_remove)

        # Remove non-serializable components
        sys_inst = env.get_wrapper_attr('sys')
        if hasattr(sys_inst, 'model'):
            setattr(sys_inst, 'model', None)
        if hasattr(sys_inst, 'solver'):
            setattr(sys_inst, 'solver', None)
        if hasattr(sys_inst, 'instance'):
            setattr(sys_inst, 'instance', None)

        # Sort related_data_sources for deterministic ordering
        related_data_sources.sort()

        return env, related_data_sources

    @classmethod
    def _deserialize(cls, env: Env, storage: TupliStorage) -> Env:
        data_providers = get_all_data_providers(env)
        for provider in data_providers.values():
            data_source = provider.data
            # We only have to prepare the Pandas and CSV data sources
            if isinstance(data_source, (PandasDataSource, CSVDataSource)):
                # load the data from the stored datasources
                ds = storage.load_artifact(data_source.data)
                data_kwargs = {'index_col': 't'}
                ds = ds.decode('utf-8')
                d = io.StringIO(ds)
                loaded_df = pd.read_csv(d, **data_kwargs)
                loaded_df.index = pd.to_datetime(loaded_df.index)
                setattr(data_source, 'data', loaded_df)
        return env

    def get_system(self) -> System:
        """Returns the system instance from the environment."""
        return self.env.get_wrapper_attr('sys')
