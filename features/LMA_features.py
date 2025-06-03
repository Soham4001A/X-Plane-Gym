import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from typing import Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#from gym import spaces as gymnasium_spaces #TESTING OLD GYM FOR RENDERING
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium import spaces as gymnasium_spaces

# Assuming f16_ils_env.py defines these, or define them here for clarity if needed
# from f16_ils_env import OBS_FEATURE_NAMES, NUM_OBS_FEATURES (if you want to import)
# For now, let's hardcode the expected input size to F16ILSFeatureExtractor

# --- Constants for F16ILSFeatureExtractor ---
# These should match the structure of a single frame from F16ILSEnv's observation buffer
# Order: delta_loc_deg, delta_gs_deg, airspeed_error_kts, vs_fps, pitch_rad, roll_rad, hdg_err_rad, alt_agl_ft, aoa_deg, p_rate, q_rate, dist_nm
F16_ILS_RAW_FEATURE_COUNT = 12 # Number of raw features in a single observation frame from the env

class F16ILSFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for a single frame of the F16ILSEnv observation.
    Transforms raw ILS-related features into a processed feature vector.
    """
    def __init__(self, observation_space: gymnasium_spaces.Box):
        # Input observation_space is for a single frame, expected to be 1D
        if len(observation_space.shape) != 1 or observation_space.shape[0] != F16_ILS_RAW_FEATURE_COUNT:
            raise ValueError(
                f"F16ILSFeatureExtractor expects a 1D observation space with {F16_ILS_RAW_FEATURE_COUNT} features, "
                f"got shape {observation_space.shape}"
            )

        # Calculate output dimension:
        # delta_loc_deg (1) -> scaled
        # delta_gs_deg (1) -> scaled
        # airspeed_error_kts (1) -> scaled
        # vertical_speed_fps (1) -> scaled
        # pitch_angle_rad (2: sin, cos)
        # roll_angle_rad (2: sin, cos)
        # heading_error_rad (2: sin, cos)
        # altitude_agl_ft (1) -> scaled
        # alpha_deg (1) -> scaled
        # pitch_rate_rad_s (1) -> scaled
        # roll_rate_rad_s (1) -> scaled
        # distance_to_threshold_nm (1) -> scaled
        features_dim = 1 + 1 + 1 + 1 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1 # = 15 features
        super().__init__(observation_space, features_dim=features_dim)

        # Scaling factors (can be tuned)
        self.scale_delta_loc_gs = 1.0 / 5.0  # Normalize roughly to +/-1 for +/-5 deg
        self.scale_airspeed_err = 1.0 / 20.0 # Normalize roughly to +/-1 for +/-20 kts
        self.scale_vs_fps = 1.0 / 50.0      # Normalize roughly to +/-1 for +/-50 fps (3000 fpm)
        self.scale_alt_agl_ft = 1.0 / 3000.0 # Normalize by typical initial approach AGL max
        self.scale_alpha_deg = 1.0 / 15.0   # Normalize by typical operational AoA range
        self.scale_ang_rate = 1.0 / (np.pi/2) # Normalize by 90 deg/s
        self.scale_dist_nm = 1.0 / 15.0     # Normalize by typical max start distance

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (Batch, F16_ILS_RAW_FEATURE_COUNT)
        
        # Unpack (assuming the order from F16ILSEnv's _get_observation)
        delta_loc_deg = observations[:, 0:1]
        delta_gs_deg = observations[:, 1:2]
        airspeed_error_kts = observations[:, 2:3]
        vs_fps = observations[:, 3:4]
        pitch_rad = observations[:, 4:5]
        roll_rad = observations[:, 5:6]
        hdg_err_rad = observations[:, 6:7]
        alt_agl_ft = observations[:, 7:8]
        alpha_deg = observations[:, 8:9]
        pitch_rate_rad_s = observations[:, 9:10]
        roll_rate_rad_s = observations[:, 10:11]
        dist_nm = observations[:, 11:12]

        # Apply scaling
        delta_loc_scaled = delta_loc_deg * self.scale_delta_loc_gs
        delta_gs_scaled = delta_gs_deg * self.scale_delta_loc_gs
        airspeed_err_scaled = airspeed_error_kts * self.scale_airspeed_err
        vs_scaled = vs_fps * self.scale_vs_fps
        alt_agl_scaled = alt_agl_ft * self.scale_alt_agl_ft
        alpha_scaled = alpha_deg * self.scale_alpha_deg
        pitch_rate_scaled = pitch_rate_rad_s * self.scale_ang_rate
        roll_rate_scaled = roll_rate_rad_s * self.scale_ang_rate
        dist_scaled = dist_nm * self.scale_dist_nm

        # Angles to Sine/Cosine pairs
        sin_pitch, cos_pitch = torch.sin(pitch_rad), torch.cos(pitch_rad)
        sin_roll, cos_roll = torch.sin(roll_rad), torch.cos(roll_rad)
        sin_hdg_err, cos_hdg_err = torch.sin(hdg_err_rad), torch.cos(hdg_err_rad)

        processed_features = torch.cat([
            delta_loc_scaled, delta_gs_scaled, airspeed_err_scaled, vs_scaled,
            sin_pitch, cos_pitch, sin_roll, cos_roll, sin_hdg_err, cos_hdg_err,
            alt_agl_scaled, alpha_scaled,
            pitch_rate_scaled, roll_rate_scaled, dist_scaled
        ], dim=1)
        
        return processed_features

# ===============================================
# LMA Core Components (Copied from your LMA_features.py)
# LayerNorm, LMAConfigRL, LMA_InitialTransform_RL, LatentAttention_RL, LatentMLP_RL, LMABlock_RL
# find_closest_divisor
# These remain unchanged as they are general LMA components.
# ===============================================

# --- Helper Function for LMA Configuration (Copied) ---
def find_closest_divisor(total_value: int, target_divisor: int, max_delta: int = 100) -> int:
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be a non-negative integer.")
    if total_value == 0: return 1
    if total_value % target_divisor == 0: return target_divisor
    search_start = target_divisor
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0: return candidate_minus
        candidate_plus = search_start + delta
        if total_value % candidate_plus == 0: return candidate_plus
    # print(f"Warning: No divisor found near {target_divisor} for {total_value} within delta={max_delta}. Searching all divisors.")
    best_divisor = 1
    min_diff = abs(target_divisor - 1)
    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1, div2 = i, total_value // i
            diff1, diff2 = abs(target_divisor - div1), abs(target_divisor - div2)
            if diff1 < min_diff: min_diff, best_divisor = diff1, div1
            if diff2 < min_diff: min_diff, best_divisor = diff2, div2
    if abs(target_divisor - total_value) < min_diff: best_divisor = total_value
    # print(f"Using {best_divisor} as fallback divisor for {total_value} (target: {target_divisor}).")
    return best_divisor

# --- LMAConfigRL (Copied) ---
@dataclass
class LMAConfigRL:
    seq_len: int; embed_dim: int; num_heads_stacking: int; target_l_new: int
    d_new: int; num_heads_latent: int
    L_new: int = field(init=False); C_new: int = field(init=False)
    def __post_init__(self):
        if not all(x > 0 for x in [self.seq_len, self.embed_dim, self.num_heads_stacking, self.target_l_new, self.d_new, self.num_heads_latent]):
            raise ValueError("All LMAConfigRL numeric inputs must be positive.")
        if self.embed_dim % self.num_heads_stacking != 0: raise ValueError(f"LMA embed_dim ({self.embed_dim}) must be divisible by num_heads_stacking ({self.num_heads_stacking}).")
        if self.d_new % self.num_heads_latent != 0: raise ValueError(f"LMA d_new ({self.d_new}) must be divisible by num_heads_latent ({self.num_heads_latent}).")
        total_features = self.seq_len * self.embed_dim
        if total_features == 0: raise ValueError("LMA total features (seq_len * embed_dim) cannot be zero.")
        try:
            self.L_new = find_closest_divisor(total_features, self.target_l_new)
            # if self.L_new != self.target_l_new: print(f"LMAConfigRL ADJUSTMENT: L_new changed from target {self.target_l_new} to {self.L_new} for total_features ({total_features}).")
            if self.L_new <= 0: raise ValueError("Calculated L_new must be positive.")
            if total_features % self.L_new != 0: raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by final L_new ({self.L_new}).")
            self.C_new = total_features // self.L_new
            if self.C_new <= 0: raise ValueError("Calculated C_new must be positive.")
        except ValueError as e: raise ValueError(f"Error in LMAConfigRL calculating L_new/C_new: {e}") from e

# --- LayerNorm (Copied) ---
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__(); self.weight = nn.Parameter(torch.ones(ndim)); self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor: return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)

# --- LMA_InitialTransform_RL (Copied) ---
class LMA_InitialTransform_RL(nn.Module):
    def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
        super().__init__(); self.lma_config = lma_config
        self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=bias)
        self.input_embedding_act = nn.ReLU(); self.embedding_dropout = nn.Dropout(p=dropout)
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=bias)
        self.embed_layer_2_act = nn.ReLU()
    def _positional_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        if L != self.lma_config.seq_len: raise ValueError(f"Input sequence length ({L}) does not match LMAConfigRL.seq_len ({self.lma_config.seq_len}).")
        y = self.input_embedding_act(self.input_embedding(x))
        y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device)
        y = self.embedding_dropout(y)
        d0 = self.lma_config.embed_dim; nh_stack = self.lma_config.num_heads_stacking; dk_stack = d0 // nh_stack
        head_views = torch.split(y, dk_stack, dim=2)
        x_stacked_seq = torch.cat(head_views, dim=1)
        x_flat = x_stacked_seq.reshape(B, -1)
        x_rechunked = x_flat.view(B, self.lma_config.L_new, self.lma_config.C_new)
        z = self.embed_layer_2_act(self.embed_layer_2(x_rechunked))
        return z

# --- LatentAttention_RL (Copied) ---
class LatentAttention_RL(nn.Module):
    def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
        super().__init__(); assert d_new % num_heads_latent == 0
        self.d_new = d_new; self.num_heads = num_heads_latent; self.head_dim = d_new // num_heads_latent
        self.c_attn = nn.Linear(d_new, 3 * d_new, bias=bias); self.c_proj = nn.Linear(d_new, d_new, bias=bias)
        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout); self.dropout_p = dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        # if self.flash: print("LatentAttention_RL: Using F.scaled_dot_product_attention.") else: print("LatentAttention_RL: Using manual attention.")
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L_new, _ = z.size(); q, k, v = self.c_attn(z).split(self.d_new, dim=2)
        q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash: y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        else:
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att_probs = F.softmax(att_scores, dim=-1); att_probs = self.attn_dropout(att_probs); y = att_probs @ v
        y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new); y = self.resid_dropout(self.c_proj(y))
        return y

# --- LatentMLP_RL (Copied) ---
class LatentMLP_RL(nn.Module):
    def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.c_fc = nn.Linear(d_new, ff_latent_hidden, bias=bias); self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ff_latent_hidden, d_new, bias=bias); self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

# --- LMABlock_RL (Copied) ---
class LMABlock_RL(nn.Module):
    def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.ln_1 = LayerNorm(lma_config.d_new, bias=bias)
        self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias)
        self.ln_2 = LayerNorm(lma_config.d_new, bias=bias)
        self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)
    def forward(self, z: torch.Tensor) -> torch.Tensor: z = z + self.attn(self.ln_1(z)); z = z + self.mlp(self.ln_2(z)); return z

# --- LMAFeaturesExtractor (Copied, logic remains general) ---
class LMAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium_spaces.Box, embed_dim: int, num_heads_stacking: int, target_l_new: int, d_new: int, num_heads_latent: int, ff_latent_hidden: int, num_lma_layers: int, seq_len: int, dropout: float, bias: bool):
        if not isinstance(observation_space, gymnasium_spaces.Box): print(f"Warning: LMAFeaturesExtractor received type {type(observation_space)}.")
        if len(observation_space.shape) != 1: raise ValueError(f"LMAFeaturesExtractor expects a 1D obs space, got {observation_space.shape}.")
        self.input_dim_total = observation_space.shape[0]; self.seq_len = seq_len
        if self.input_dim_total % seq_len != 0: raise ValueError(f"Total input dim ({self.input_dim_total}) must be divisible by seq_len ({seq_len}).")
        self.features_per_step = self.input_dim_total // seq_len
        try: _lma_config_temp = LMAConfigRL(seq_len=self.seq_len, embed_dim=embed_dim, num_heads_stacking=num_heads_stacking, target_l_new=target_l_new, d_new=d_new, num_heads_latent=num_heads_latent)
        except ValueError as e: raise ValueError(f"Failed to init LMAConfigRL for LMAFeaturesExtractor: {e}") from e
        feature_dim_out = _lma_config_temp.L_new * _lma_config_temp.d_new
        super().__init__(observation_space, features_dim=feature_dim_out)
        self.lma_config = _lma_config_temp
        self.initial_transform = LMA_InitialTransform_RL(features_per_step=self.features_per_step, lma_config=self.lma_config, dropout=dropout, bias=bias)
        self.lma_blocks = nn.ModuleList([LMABlock_RL(lma_config=self.lma_config, ff_latent_hidden=ff_latent_hidden, dropout=dropout, bias=bias) for _ in range(num_lma_layers)])
        self.flatten = nn.Flatten()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]; x_reshaped = observations.view(batch_size, self.seq_len, self.features_per_step)
        z = self.initial_transform(x_reshaped);
        for block in self.lma_blocks: z = block(z)
        features = self.flatten(z); return features

# ===============================================
# Stacked LMA Feature Extractor for F16 ILS
# ===============================================
class StackedLMA_ILS_FeaturesExtractor(BaseFeaturesExtractor):
    """
    A two-stage features extractor for F16ILSEnv with stacked frame observations.
    1. Applies F16ILSFeatureExtractor to each frame.
    2. Processes the sequence of these extracted features using LMAFeaturesExtractor.

    Input observation_space: 2D Box (num_stacked_frames, raw_features_per_frame_from_env).
    Hyperparameters for LMA (lma_*) are passed to the internal LMAFeaturesExtractor.
    """
    def __init__(self,
                 observation_space: gymnasium_spaces.Box, # This is the env's observation_space
                 lma_embed_dim_d0: int = 64,
                 lma_num_heads_stacking: int = 4,
                 lma_num_heads_latent: int = 4,
                 lma_ff_latent_hidden: int = 128,
                 lma_num_layers: int = 2,
                 lma_dropout: float = 0.1,
                 lma_bias: bool = True):

        assert isinstance(observation_space, GymnasiumBox), \
            f"Expected gymnasium.spaces.Box, got {type(observation_space)}"
        assert len(observation_space.shape) == 2, \
            "Observation space for StackedLMA_ILS_FeaturesExtractor must be 2D"

        self.env_num_stacked_frames = observation_space.shape[0]
        self.env_raw_obs_dim_per_frame = observation_space.shape[1] # Should be F16_ILS_RAW_FEATURE_COUNT (12)

        # 1. Per-frame extractor (F16ILSFeatureExtractor)
        # Create a dummy 1D observation space for a single frame to initialize F16ILSFeatureExtractor
        _single_frame_gym_space = gymnasium_spaces.Box(
            low=observation_space.low[0],
            high=observation_space.high[0],
            shape=(self.env_raw_obs_dim_per_frame,),
            dtype=observation_space.dtype
        )
        _temp_ils_extractor = F16ILSFeatureExtractor(_single_frame_gym_space)
        self.features_per_frame_after_ils_proc = _temp_ils_extractor.features_dim # e.g., 15
        del _temp_ils_extractor

        # 2. LMA Extractor setup
        # Input to LMA is a sequence of features_per_frame_after_ils_proc
        _lma_input_seq_len_L = self.env_num_stacked_frames # This is `seq_len` for LMAFeaturesExtractor
        
        # LMAFeaturesExtractor expects a flat 1D observation space
        _lma_total_input_features_flat = _lma_input_seq_len_L * self.features_per_frame_after_ils_proc
        
        _dummy_lma_flat_obs_space = gymnasium_spaces.Box(
            low=-np.inf, high=np.inf, shape=(_lma_total_input_features_flat,), dtype=observation_space.dtype
        )

        # Determine `target_l_new` and `d_new` for LMA based on its input.
        # These can be made configurable or derived. Let's use simple defaults as in your original.
        # Example: target_l_new could be _lma_input_seq_len_L // 2 or a fixed value
        # For consistency with your main script, let's assume they are passed or defaulted
        _lma_target_l_new = _lma_input_seq_len_L // 2 if _lma_input_seq_len_L > 1 else 1 # Or from args
        _lma_target_d_new = lma_embed_dim_d0 // 2 if lma_embed_dim_d0 > 1 else lma_embed_dim_d0 # Or from args


        _temp_lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space,
            embed_dim=lma_embed_dim_d0,
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_new, # Target L' for LMA
            d_new=_lma_target_d_new, # D' for LMA
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=_lma_input_seq_len_L, # L for LMA (env_num_stacked_frames)
            dropout=lma_dropout,
            bias=lma_bias
        )
        final_features_dim = _temp_lma_extractor.features_dim
        del _temp_lma_extractor

        super().__init__(observation_space, features_dim=final_features_dim)

        # Actual module initializations
        self.ils_feature_extractor = F16ILSFeatureExtractor(_single_frame_gym_space)
        
        self.lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space,
            embed_dim=lma_embed_dim_d0,
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_new,
            d_new=_lma_target_d_new,
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=_lma_input_seq_len_L,
            dropout=lma_dropout,
            bias=lma_bias
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        obs_reshaped_for_ils_proc = observations.reshape(
            batch_size * self.env_num_stacked_frames,
            self.env_raw_obs_dim_per_frame
        )
        
        extracted_features_ils = self.ils_feature_extractor(obs_reshaped_for_ils_proc)
        
        processed_obs_for_lma_flat = extracted_features_ils.reshape(
            batch_size,
            self.env_num_stacked_frames * self.features_per_frame_after_ils_proc
        )
        
        final_features = self.lma_extractor(processed_obs_for_lma_flat)
        return final_features