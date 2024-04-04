from abc import ABCMeta

from pydantic import BaseModel, ConfigDict


class SB3AlgorithmBaseConfig(BaseModel):
    policy: str = 'MlpPolicy'
    device: str = 'cpu'
    batch_size: int = 12
    learning_rate: float = 0.0003


class SB3PPOConfig(SB3AlgorithmBaseConfig):
    n_steps: int = 24
    use_sde: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 0.9
    normalize_advantage: bool = True
    policy_kwargs: dict = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=0, squash_output=True)


class SB3SACConfig(SB3AlgorithmBaseConfig):
    buffer_size: int = 1000000
    learning_starts: int = 100
    train_freq: int = 24
    policy_kwargs: dict = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))


class SB3BaseConfig(BaseModel):
    total_steps: int
    algorithm: ABCMeta
    seed: int
    algorithm_config: SB3AlgorithmBaseConfig
    penalty_factor: float = 0.0
    # necessary for ABCMeta type
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MAPPOBaseConfig(BaseModel):
    algorithm_name: str
    seed: int
    num_env_steps: int
    cuda: bool = False
    cuda_deterministic: bool = True
    n_training_threads: int = 1
    n_rollout_threads: int = 1
    n_eval_rollout_threads: int = 1
    episode_length: int = 24
    share_policy: bool = False
    use_centralized_V: bool = True
    hidden_size: int = 64
    layer_N: int = 1
    use_ReLU: bool = True
    use_popart: bool = False
    use_valuenorm: bool = True
    use_feature_normalization: bool = False
    use_orthogonal: bool = True
    gain: float = 0.01
    use_naive_recurrent_policy: bool = False
    use_recurrent_policy: bool = False
    recurrent_N: int = 1
    data_chunk_length: int = 10
    lr: float = 0.0005
    critic_lr: float = 0.0005
    opti_eps: float = 1e-05
    weight_decay: float = 0.0
    ppo_epoch: int = 15
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    use_max_grad_norm: bool = True
    max_grad_norm: float = 10.0
    use_gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    stacked_frames: int = 1
    use_proper_time_limits: bool = False
    use_huber_loss: bool = True
    use_value_active_masks: bool = False  # we do not need masks since no agents terminate prematurely
    use_policy_active_masks: bool = False  # we do not need masks since no agents terminate prematurely
    huber_delta: float = 10.0
    use_linear_lr_decay: bool = False
    log_interval: int = 1
    use_eval: bool = False
    eval_interval: int = 25
    eval_episodes: int = 32
    ifi: float = 0.1
    # args from Commonpower
    penalty_factor: float = 0.0
