import warnings
from typing import Any, ClassVar, Dict, Optional, TypeVar, Union # Added Dict

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, get_device

# For AM-PPO custom optimizer
try:
    from torch.optim import Adam
    from stable_baselines3.ppo.optim.sgd import AlphaGrad, DAG
except ImportError:
    warnings.warn("AM-PPO custom optimizers (AlphaGrad, DAG) not found. Only Adam will be available for AM-PPO.")
    AlphaGrad = None
    DAG = None


SelfPPO = TypeVar("SelfPPO", bound="PPO")

# ### AGRAM Controller Function (copied from CleanRL AMPPO.py) ###
# Ensure this function is defined within the scope of the PPO class or imported
def dynago_transform_advantages(
    raw_advantages_batch: th.Tensor,
    dynago_params_A: Dict,
    alpha_A_ema_state: th.Tensor,
    prev_saturation_A_ema_state: th.Tensor,
    dynago_kappa_for_formula: float, # Added to pass the specific kappa for the formula
    dynago_v_shift_for_formula: float, # Added for v_shift
    update_ema: bool = True,
) -> th.Tensor:
    current_raw_advantages_MB = raw_advantages_batch

    if current_raw_advantages_MB.numel() <= 1: # Or even 0 if batch size is 1 and advantages are single value
        return current_raw_advantages_MB.clone() if current_raw_advantages_MB.numel() > 0 else th.tensor(0.0, device=current_raw_advantages_MB.device)


    kappa_controller = dynago_params_A["kappa"] # Kappa for controller's target alpha
    tau = dynago_params_A["tau"]
    p_star = dynago_params_A["p_star"]
    eta = dynago_params_A["eta"]
    rho = dynago_params_A["rho"]
    eps = dynago_params_A["eps"]
    alpha_min = dynago_params_A["alpha_min"]
    alpha_max = dynago_params_A["alpha_max"]
    rho_sat = dynago_params_A["rho_sat"]

    N_A = th.linalg.norm(current_raw_advantages_MB)
    # if N_A < eps: # If norm is too small, original advantages might be near zero.
    #     return current_raw_advantages_MB # Return original to avoid division by zero or instability

    # Ensure sigma_A is reasonably well-behaved, especially for small batches
    # For very small numel, std can be 0 or NaN.
    if current_raw_advantages_MB.numel() > 1:
        sigma_A = th.std(current_raw_advantages_MB) + eps
    else: # Single element, std is not well-defined or 0. Use a fallback.
        sigma_A = th.abs(current_raw_advantages_MB.item()) + eps if current_raw_advantages_MB.numel() == 1 else eps


    alpha_A_prev_ema_val = alpha_A_ema_state.clone() # Use .clone() to avoid in-place modification issues if not intended
    prev_sat_A_ema_val = prev_saturation_A_ema_state.clone()

    alpha_A_hat = (
        kappa_controller
        * (N_A + eps) / (sigma_A + eps) # Ensure sigma_A is not zero
        * (p_star / (prev_sat_A_ema_val + eps)) ** eta
    )
    # Ensure alpha_A_hat is a scalar tensor if inputs are
    if not isinstance(alpha_A_hat, th.Tensor): alpha_A_hat = th.tensor(alpha_A_hat, device=N_A.device)
    if alpha_A_hat.ndim > 0 : alpha_A_hat = alpha_A_hat.mean() # Ensure scalar

    alpha_A_to_use_for_Z = None
    if update_ema:
        _alpha_A_updated = (1 - rho) * alpha_A_prev_ema_val[0] + rho * alpha_A_hat
        _alpha_A_updated = th.clamp(_alpha_A_updated, alpha_min, alpha_max)
        alpha_A_ema_state[0] = _alpha_A_updated.detach() 
        alpha_A_to_use_for_Z = _alpha_A_updated
    else:
        alpha_A_to_use_for_Z = alpha_A_prev_ema_val[0] 

    normalized_advantages_A_MB = current_raw_advantages_MB / (N_A + eps) # Ensure N_A is not zero
    Z_A_MB = alpha_A_to_use_for_Z * normalized_advantages_A_MB

    if update_ema:
        current_observed_saturation_A = (Z_A_MB.abs() > tau).float().mean()
        prev_saturation_A_ema_state[0] = (
            (1 - rho_sat) * prev_sat_A_ema_val[0] + rho_sat * current_observed_saturation_A
        ).detach()

    modulation_factor = (dynago_kappa_for_formula * th.tanh(Z_A_MB) + dynago_v_shift_for_formula) # Use passed kappa and v_shift
    modulated_advantages_MB = th.abs(current_raw_advantages_MB) * modulation_factor
    
    return modulated_advantages_MB


class PPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True, # SB3's default normalizes raw advantages
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
        # AM-PPO / AGRAM specific arguments
        use_am_ppo: bool = False,
        am_ppo_optimizer: str = "Adam", # For AM-PPO: Adam, AlphaGrad, DAG
        am_ppo_alpha_optimizer: float = 0.0, # For AlphaGrad
        dynago_tau: float = 1.25,
        dynago_p_star: float = 0.10,
        dynago_kappa: float = 2.0, # This is kappa_formula in CleanRL args
        dynago_eta: float = 0.3,
        dynago_rho: float = 0.1,
        dynago_eps: float = 1e-5,
        dynago_alpha_min: float = 1e-12,
        dynago_alpha_max: float = 1e12,
        dynago_rho_sat: float = 0.98,
        dynago_alpha_A_init: float = 1.0,
        dynago_prev_sat_A_init: float = 0.10,
        dynago_v_shift: float = 0.0, # As per CleanRL, effectively 0
        am_ppo_norm_adv: bool = True, # Corresponds to args.norm_adv in CleanRL (normalize modulated adv)
        dynago_kappa_controller: Optional[float] = None, # Optional: kappa for controller, if different from formula kappa

    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False, # Will call custom _setup_model
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage: # This is SB3's original normalize_advantage for raw GAE
            assert batch_size > 1, "`batch_size` must be greater than 1 if using SB3's normalize_advantage."
        
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if use_am_ppo and am_ppo_norm_adv: # AM-PPO's norm_adv is on minibatch
                 pass # Check happens per minibatch later
            elif normalize_advantage : # SB3's original check
                assert buffer_size > 1, f"`n_steps * n_envs` must be greater than 1 for SB3's normalize_advantage."

            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"Mini-batch size {batch_size} does not evenly divide RolloutBuffer size {buffer_size}."
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage # SB3's original flag
        self.target_kl = target_kl

        # AM-PPO specific attributes
        self.use_am_ppo = use_am_ppo
        if self.use_am_ppo:
            self.am_ppo_optimizer_name = am_ppo_optimizer.lower()
            self.am_ppo_alpha_optimizer = am_ppo_alpha_optimizer
            self.dynago_params_A_config = {
                "kappa": dynago_kappa_controller if dynago_kappa_controller is not None else dynago_kappa, # Controller kappa
                "tau": dynago_tau, "p_star": dynago_p_star, "eta": dynago_eta,
                "rho": dynago_rho, "eps": dynago_eps, "alpha_min": dynago_alpha_min,
                "alpha_max": dynago_alpha_max, "rho_sat": dynago_rho_sat,
            }
            self.dynago_kappa_formula = dynago_kappa # Kappa for the tanh modulation formula
            self.dynago_v_shift_formula = dynago_v_shift # v_shift for the tanh modulation formula
            
            # Ensure these are initialized on the correct device later in _setup_model
            self.alpha_A_ema_state = th.tensor([dynago_alpha_A_init], dtype=th.float32)
            self.prev_saturation_A_ema_state = th.tensor([dynago_prev_sat_A_init], dtype=th.float32)
            self.am_ppo_norm_adv = am_ppo_norm_adv # Whether to normalize AGRAM's modulated advantages

            if verbose > 0:
                print("AM-PPO extensions enabled.")
                print(f"  AM-PPO Optimizer: {self.am_ppo_optimizer_name}")
                print(f"  AM-PPO Normalize Modulated Advantages: {self.am_ppo_norm_adv}")


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # Call SB3's original _setup_model first
        super()._setup_model()

        # Initialize schedules for policy/value clipping (standard PPO)
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive."
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if self.use_am_ppo:
            # Move AGRAM EMA states to the correct device
            self.alpha_A_ema_state = self.alpha_A_ema_state.to(self.device)
            self.prev_saturation_A_ema_state = self.prev_saturation_A_ema_state.to(self.device)

            # Setup AM-PPO optimizer if specified
            if self.am_ppo_optimizer_name == "adam":
                self.policy.optimizer = Adam(self.policy.parameters(), lr=self.lr_schedule(1), eps=1e-5) # Default Adam eps
                if self.verbose > 0: print(f"AM-PPO using Adam optimizer with LR: {self.lr_schedule(1)}")
            elif self.am_ppo_optimizer_name == "alphagrad" and AlphaGrad is not None:
                self.policy.optimizer = AlphaGrad(self.policy.parameters(), lr=self.lr_schedule(1), alpha=self.am_ppo_alpha_optimizer)
                if self.verbose > 0: print(f"AM-PPO using AlphaGrad optimizer with LR: {self.lr_schedule(1)}, Alpha: {self.am_ppo_alpha_optimizer}")
            elif self.am_ppo_optimizer_name == "dag" and DAG is not None:
                self.policy.optimizer = DAG(self.policy.parameters(), lr=self.lr_schedule(1))
                if self.verbose > 0: print(f"AM-PPO using DAG optimizer with LR: {self.lr_schedule(1)}")
            else:
                if self.verbose > 0:
                    print(f"AM-PPO optimizer '{self.am_ppo_optimizer_name}' not recognized or unavailable, "
                          f"falling back to Adam setup by SB3.")
                # If policy.optimizer wasn't created by super() or needs specific Adam for AM-PPO:
                if not hasattr(self.policy, 'optimizer') or not isinstance(self.policy.optimizer, optim.Adam):
                     self.policy.optimizer = Adam(self.policy.parameters(), lr=self.lr_schedule(1), eps=1e-5)


    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions = []
        
        # For AM-PPO logging
        am_ppo_mean_raw_adv_epoch_list = []
        am_ppo_mean_mod_adv_epoch_list = []

        continue_training = True

        # AM-PPO: Update AGRAM EMAs once per iteration, using the full batch of raw advantages
        if self.use_am_ppo:
            # self.rollout_buffer.advantages is a NumPy array
            all_raw_advantages_np = self.rollout_buffer.advantages.flatten() # This is still a NumPy array

            # Check size of NumPy array
            if all_raw_advantages_np.size > 0 :
                # Convert NumPy array to PyTorch tensor and move to device before passing
                all_raw_advantages_th = th.tensor(all_raw_advantages_np, dtype=th.float32).to(self.device)
                
                _ = dynago_transform_advantages(
                    raw_advantages_batch=all_raw_advantages_th, # Pass the tensor
                    dynago_params_A=self.dynago_params_A_config,
                    alpha_A_ema_state=self.alpha_A_ema_state,
                    prev_saturation_A_ema_state=self.prev_saturation_A_ema_state,
                    dynago_kappa_for_formula=self.dynago_kappa_formula,
                    dynago_v_shift_for_formula=self.dynago_v_shift_formula,
                    update_ema=True # Update EMAs
                )
            else:
                if self.verbose > 0: print("AM-PPO: Skipping EMA update due to empty advantages batch.")

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # For AM-PPO logging per epoch
            epoch_raw_advs_mb_means = []
            epoch_mod_advs_mb_means = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                # Advantages from buffer (raw GAE)
                advantages_raw_mb = rollout_data.advantages # These are per-minibatch raw GAE

                if self.use_am_ppo:
                    if advantages_raw_mb.numel() > 0:
                        epoch_raw_advs_mb_means.append(advantages_raw_mb.mean().item())
                        # Modulate advantages for policy loss (EMA states are frozen during epoch)
                        advantages_modulated_mb = dynago_transform_advantages(
                            raw_advantages_batch=advantages_raw_mb,
                            dynago_params_A=self.dynago_params_A_config,
                            alpha_A_ema_state=self.alpha_A_ema_state, # Use current (frozen for epoch) EMA
                            prev_saturation_A_ema_state=self.prev_saturation_A_ema_state, # Use current
                            dynago_kappa_for_formula=self.dynago_kappa_formula,
                            dynago_v_shift_for_formula=self.dynago_v_shift_formula,
                            update_ema=False # Do NOT update EMAs within epoch minibatch processing
                        )
                        epoch_mod_advs_mb_means.append(advantages_modulated_mb.mean().item())
                    else: # Handle case of empty advantages minibatch
                        advantages_modulated_mb = advantages_raw_mb.clone() # Or some other default
                        if advantages_raw_mb.numel() > 0: epoch_raw_advs_mb_means.append(0.0) # Should not happen if numel > 0
                        if advantages_modulated_mb.numel() > 0: epoch_mod_advs_mb_means.append(0.0)


                    # AM-PPO uses modulated advantages for policy loss.
                    # Optionally normalize these modulated advantages.
                    if self.am_ppo_norm_adv and advantages_modulated_mb.numel() > 1:
                        advantages_for_policy = (advantages_modulated_mb - advantages_modulated_mb.mean()) / (advantages_modulated_mb.std() + 1e-8)
                    else:
                        advantages_for_policy = advantages_modulated_mb
                else: # Standard PPO
                    advantages_for_policy = advantages_raw_mb
                    if self.normalize_advantage and advantages_for_policy.numel() > 1: # SB3's original normalization
                        advantages_for_policy = (advantages_for_policy - advantages_for_policy.mean()) / (advantages_for_policy.std() + 1e-8)
                
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages_for_policy * ratio
                policy_loss_2 = advantages_for_policy * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.use_am_ppo:
                    target_values = advantages_modulated_mb + rollout_data.old_values 
                else:
                    # Standard PPO: uses TD(gae_lambda) target (rollout_data.returns)
                    target_values = rollout_data.returns

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                value_loss = F.mse_loss(target_values, values_pred) # Use AM-PPO or standard target_values
                value_losses.append(value_loss.item())

                if entropy is None: entropy_loss = -th.mean(-log_prob)
                else: entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False; break
                
                self.policy.optimizer.zero_grad(); loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            if self.use_am_ppo:
                if epoch_raw_advs_mb_means: am_ppo_mean_raw_adv_epoch_list.append(np.mean(epoch_raw_advs_mb_means))
                if epoch_mod_advs_mb_means: am_ppo_mean_mod_adv_epoch_list.append(np.mean(epoch_mod_advs_mb_means))

            self._n_updates += 1
            if not continue_training: break
        
        # Standard PPO Logging
        explained_var_values_np = self.rollout_buffer.values.flatten()
        if self.use_am_ppo:
            all_raw_adv_for_exp_var_np = self.rollout_buffer.advantages.flatten()
            if all_raw_adv_for_exp_var_np.size > 0: 
                all_raw_adv_for_exp_var_th = th.tensor(all_raw_adv_for_exp_var_np, dtype=th.float32).to(self.device)
                all_mod_adv_for_exp_var = dynago_transform_advantages(
                    all_raw_adv_for_exp_var_th, self.dynago_params_A_config,
                    self.alpha_A_ema_state, self.prev_saturation_A_ema_state,
                    self.dynago_kappa_formula, self.dynago_v_shift_formula, update_ema=False
                )
                # Ensure self.rollout_buffer.values is also a tensor and on the same device for addition
                values_for_exp_var_th = th.tensor(self.rollout_buffer.values.flatten(), dtype=th.float32).to(self.device)
                explained_var_returns = (all_mod_adv_for_exp_var + values_for_exp_var_th).cpu()
            else: # Fallback if advantages are empty
                explained_var_returns = th.tensor(self.rollout_buffer.returns.flatten(), dtype=th.float32) # ensure tensor

        else:
            explained_var_returns = th.tensor(self.rollout_buffer.returns.flatten(), dtype=th.float32) # ensure tensor
        
        # Convert inputs to explained_variance to NumPy arrays
        y_pred_np = th.tensor(explained_var_values_np, dtype=th.float32).cpu().numpy()
        y_true_np = explained_var_returns.cpu().numpy()

        explained_var = explained_variance(
            y_pred_np, # Now a NumPy array
            y_true_np  # Now a NumPy array
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"): self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None: self.logger.record("train/clip_range_vf", clip_range_vf)

        if self.use_am_ppo:
            self.logger.record("am_ppo/alpha_A_ema", self.alpha_A_ema_state.item())
            self.logger.record("am_ppo/prev_saturation_A_ema", self.prev_saturation_A_ema_state.item())
            if am_ppo_mean_raw_adv_epoch_list: self.logger.record("am_ppo/mean_raw_adv_iter", np.mean(am_ppo_mean_raw_adv_epoch_list))
            if am_ppo_mean_mod_adv_epoch_list: self.logger.record("am_ppo/mean_mod_adv_iter", np.mean(am_ppo_mean_mod_adv_epoch_list))


    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        if self.use_am_ppo and tb_log_name == "PPO": # Default tb_log_name
            tb_log_name = "AM_PPO" 

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )