import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union, Callable

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, get_schedule_fn
from stable_baselines3.common.callbacks import BaseCallback # Added for type hinting

# Assuming PPO_BASIC is the standard PPO class from stable-baselines3
# For clarity, I'll rename it to PPO for inheritance.
# If PPO_BASIC is exactly stable_baselines3.ppo.PPO, then:
from ppo.ppo_BASIC import PPO_BASIC 

SelfHGRPO = TypeVar("SelfHGRPO", bound="HGRPO")

class HGRPO(PPO_BASIC): # Inherit from the provided PPO class
    """
    Hybrid Group Relative Policy Optimization (HGRPO)

    This algorithm extends PPO by incorporating multi-sample reward estimation
    for advantage calculation, as discussed.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param n_reward_samples: Number of actions to sample per state for HGRPO reward estimation.
    :param reward_fn: A callable function `reward_fn(observations, actions) -> rewards`
                      that returns the empirical reward for given state-action pairs.
                      Observations and actions are PyTorch tensors on the policy's device.
                      It should return a 1D tensor of rewards.
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE
    :param clip_range: Clipping parameter for PPO
    :param clip_range_vf: Clipping parameter for the value function
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param rollout_buffer_class: Rollout buffer class to use.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param target_kl: Limit the KL divergence between updates
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: the log location for tensorboard
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...)
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        n_reward_samples: int, # New HGRPO parameter
        reward_fn: Callable[[th.Tensor, th.Tensor], th.Tensor], # New HGRPO parameter
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model, # PPO's _init_setup_model will be called
        )
        
        self.n_reward_samples = n_reward_samples
        if not callable(reward_fn):
            raise ValueError("`reward_fn` must be a callable function.")
        self.reward_fn = reward_fn
        self.reward_norm_eps = 1e-8 # Epsilon for reward normalization

    def collect_rollouts(
        self,
        env: GymEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free evaluation of the current policy
        for a fixed number of steps across possibly multiple environments.
        This method overrides the PPO/OnPolicyAlgorithm method to inject HGRPO reward calculation.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if training should continue, False if callback requested training to stop
        """
        # Let the parent class handle the actual environment interaction and buffer filling.
        # This also calls the initial rollout_buffer.compute_returns_and_advantage based on original env rewards.
        continue_training = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        if not continue_training:
            return False

        # HGRPO: Modify rewards in the buffer and re-compute advantages/returns
        
        # Determine which part of the buffer was filled in the last rollout
        if rollout_buffer.full:
            # If buffer is full, all n_steps were collected for each env
            # The buffer stores (n_steps, n_envs, *feature_dim)
            # We process all observations in the buffer
            num_valid_steps = rollout_buffer.buffer_size # n_steps
            # obs_data_np needs to be (n_steps, n_envs, *obs_shape)
            obs_data_np = rollout_buffer.observations 
        else:
            # Buffer is not full, data is up to rollout_buffer.pos
            num_valid_steps = rollout_buffer.pos
            obs_data_np = rollout_buffer.observations[:num_valid_steps]

        # Reshape observations to (num_steps_total, *obs_shape) for batch processing
        # where num_steps_total = num_valid_steps * self.n_envs
        original_obs_shape = obs_data_np.shape
        if self.n_envs > 1:
            # (num_valid_steps, n_envs, *obs_dim) -> (num_valid_steps * n_envs, *obs_dim)
            obs_data_np_flat = obs_data_np.reshape(num_valid_steps * self.n_envs, *self.observation_space.shape)
        else:
            # (num_valid_steps, *obs_dim) -> (num_valid_steps, *obs_dim)
            obs_data_np_flat = obs_data_np.reshape(num_valid_steps, *self.observation_space.shape)
        
        obs_tensor = self.rollout_buffer.to_torch(obs_data_np_flat)

        with th.no_grad():
            # Expand observations for N_samples: (total_steps, *obs_dim) -> (total_steps * N, *obs_dim)
            expanded_obs_tensor = obs_tensor.repeat_interleave(self.n_reward_samples, dim=0)
            
            # Sample N actions for each observation using the current policy
            # self.policy.predict returns numpy arrays by default, convert to tensor for reward_fn
            # Need to ensure predict is in non-deterministic mode.
            actions_for_reward_estimation_np, _ = self.policy.predict(expanded_obs_tensor.cpu().numpy(), deterministic=False)
            actions_for_reward_estimation_tensor = self.rollout_buffer.to_torch(actions_for_reward_estimation_np)
            
            # Get empirical rewards using the user-provided reward_fn
            # Expected output shape: (total_steps * N_samples,)
            raw_rewards_expanded = self.reward_fn(expanded_obs_tensor, actions_for_reward_estimation_tensor)
            
            if raw_rewards_expanded.shape[0] != expanded_obs_tensor.shape[0]:
                raise ValueError(
                    f"reward_fn returned tensor of shape {raw_rewards_expanded.shape}, "
                    f"expected ({expanded_obs_tensor.shape[0]},)"
                )

            # Reshape raw_rewards to (total_steps, N_samples) for normalization
            raw_rewards_N_samples = raw_rewards_expanded.reshape(obs_tensor.shape[0], self.n_reward_samples)

            # Normalize rewards per state (across N_samples) and apply tanh
            mean_rewards = raw_rewards_N_samples.mean(dim=1, keepdim=True)
            std_rewards = raw_rewards_N_samples.std(dim=1, keepdim=True) + self.reward_norm_eps
            normalized_rewards = (raw_rewards_N_samples - mean_rewards) / std_rewards
            transformed_rewards_N_samples = th.tanh(normalized_rewards)
            
            # Average transformed rewards to get Ř_t_avg for each state
            # Shape: (total_steps,)
            r_t_avg_for_buffer = transformed_rewards_N_samples.mean(dim=1)
        
        # Reshape Ř_t_avg back to (num_valid_steps, n_envs) to store in buffer
        new_rewards_np = r_t_avg_for_buffer.cpu().numpy()
        rewards_to_store_in_buffer = new_rewards_np.reshape(num_valid_steps, self.n_envs)
        
        # Replace original rewards in the buffer
        if rollout_buffer.full:
            rollout_buffer.rewards[:] = rewards_to_store_in_buffer
        else:
            rollout_buffer.rewards[:num_valid_steps] = rewards_to_store_in_buffer
            
        # Re-compute GAE and returns with these new HGRPO rewards
        # Need last_values for the state *after* the rollout ends.
        # These are computed based on self._last_obs and current policy.
        with th.no_grad():
            obs_tensor_last = self.rollout_buffer.to_torch(self._last_obs) # self._last_obs is (n_envs, *obs_shape)
            
            # Get value of last observation
            features_last = self.policy.extract_features(obs_tensor_last)
            if self.policy.share_features_extractor: # Standard SB3 policy structure
                 _, latent_vf_last = self.policy.mlp_extractor(features_last)
            else: # Separate actor/critic MLPs
                # Assume mlp_extractor.forward_critic(features) exists or adapt as needed
                # For SB3 default ActorCriticPolicy, mlp_extractor is shared.
                # If using custom policy, this might need adjustment.
                # This path might not be hit if share_features_extractor is True by default in ActorCriticPolicy
                latent_pi_last, latent_vf_last = self.policy.mlp_extractor(features_last)

            last_values_hgrpo = self.policy.value_net(latent_vf_last) # Shape (n_envs, 1)

        # self.dones comes from the end of the rollout collected by super().collect_rollouts
        rollout_buffer.compute_returns_and_advantage(last_values=last_values_hgrpo, dones=self.dones)
        
        return continue_training

    def learn(
        self: SelfHGRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HGRPO", # Changed default log name
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHGRPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )