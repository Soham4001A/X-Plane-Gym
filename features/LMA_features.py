# features/LMA_features.py

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
# from typing import Type # Not used in this snippet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box as GymnasiumBox # Use directly
from gymnasium import spaces as gymnasium_spaces # For other space types if needed

# --- Constants for XPlaneILSFeatureExtractor ---
# This should match XPlaneILSEnv.NUM_OBS_FEATURES
XPLANE_ILS_RAW_FEATURE_COUNT = 12 # Number of raw features in a single observation frame from XPlaneILSEnv

class XPlaneILSFeatureExtractor(BaseFeaturesExtractor): # Renamed for clarity
    """
    Feature extractor for a single frame of the XPlaneILSEnv observation.
    The XPlaneILSEnv already provides normalized features. This extractor can
    perform additional minimal processing if needed (e.g., sin/cos for any
    raw angles if they were passed, or just act as a pass-through/reshaper).

    Input features from XPlaneILSEnv (already normalized):
    0: norm_roll_deg
    1: norm_pitch_deg
    2: norm_heading_error_deg
    3: norm_speed_error_mps
    4: norm_aoa_deg
    5: norm_P_dps (roll rate)
    6: norm_Q_dps (pitch rate)
    7: norm_R_dps (yaw rate)
    8: norm_alt_agl_m
    9: norm_lat_dev_m
    10: norm_vert_dev_m
    11: norm_dist_to_thresh_horiz_m
    """
    def __init__(self, observation_space: gymnasium_spaces.Box):
        if len(observation_space.shape) != 1 or observation_space.shape[0] != XPLANE_ILS_RAW_FEATURE_COUNT:
            raise ValueError(
                f"XPlaneILSFeatureExtractor expects a 1D observation space with {XPLANE_ILS_RAW_FEATURE_COUNT} features, "
                f"got shape {observation_space.shape}"
            )

        # Since features are already normalized, we primarily pass them through.
        # If we wanted to do sin/cos on, say, a hypothetical raw pitch/roll, this is where it would happen.
        # For the current XPlaneILSEnv output, all 12 features are arguably fine as is.
        features_dim = XPLANE_ILS_RAW_FEATURE_COUNT # Outputting 12 features
        super().__init__(observation_space, features_dim=features_dim)

        # No scaling needed here as XPlaneILSEnv handles normalization.
        # If any feature needed specific transformation (e.g. non-linear scaling beyond simple normalization),
        # it could be done here.

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (Batch, XPLANE_ILS_RAW_FEATURE_COUNT)
        # Given the pre-normalization in XPlaneILSEnv, this can be a direct pass-through.
        # If specific features needed further transformation (e.g., log scaling for distance,
        # or sin/cos if the env provided raw cyclical angles), it would be done here.
        # For now, assuming the env's normalization is sufficient for this stage.
        
        # Example: if feature index 0 was raw roll in radians and we wanted sin/cos:
        # raw_roll_rad = observations[:, 0:1]
        # sin_roll, cos_roll = torch.sin(raw_roll_rad), torch.cos(raw_roll_rad)
        # other_features = observations[:, 1:]
        # return torch.cat([sin_roll, cos_roll, other_features], dim=1) 
        # (This would change features_dim in __init__)

        return observations # Pass-through for now
# ===============================================
# LMA Core Components (Copied from your LMA_features.py)
# LayerNorm, LMAConfigRL, LMA_InitialTransform_RL, LatentAttention_RL, LatentMLP_RL, LMABlock_RL
# find_closest_divisor
# These remain unchanged.
# ===============================================
# --- Helper Function for LMA Configuration (Copied) ---
def find_closest_divisor(total_value: int, target_divisor: int, max_delta: int = 100) -> int:
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be a non-negative integer.")
    if total_value == 0: return 1 # Should not happen if total_value check above is good
    if total_value % target_divisor == 0: return target_divisor
    # Check for target_divisor being 1, as it's a common edge case for small total_values
    if target_divisor == 1 and total_value % 1 == 0: return 1 # any total_value is divisible by 1
    
    search_start = target_divisor
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0: return candidate_minus
        candidate_plus = search_start + delta
        if total_value % candidate_plus == 0: return candidate_plus # Check for total_value % 0 not needed due to candidate_plus > 0
    # Fallback search (as before)
    best_divisor = 1
    min_diff = abs(target_divisor - 1)
    # Also check total_value itself as a divisor
    if abs(target_divisor - total_value) < min_diff:
        min_diff = abs(target_divisor - total_value)
        best_divisor = total_value

    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1, div2 = i, total_value // i
            diff1, diff2 = abs(target_divisor - div1), abs(target_divisor - div2)
            if diff1 < min_diff: min_diff, best_divisor = diff1, div1
            elif diff1 == min_diff: best_divisor = max(best_divisor, div1) # Prefer larger if diff is same
            
            if diff2 < min_diff: min_diff, best_divisor = diff2, div2
            elif diff2 == min_diff: best_divisor = max(best_divisor, div2)

    # print(f"Warning: No divisor found near {target_divisor} for {total_value} within delta={max_delta}. Using fallback {best_divisor}.")
    return best_divisor

# --- LMAConfigRL (Copied) ---
@dataclass
class LMAConfigRL:
    seq_len: int; embed_dim: int; num_heads_stacking: int; target_l_new: int
    d_new: int; num_heads_latent: int
    L_new: int = field(init=False); C_new: int = field(init=False)
    def __post_init__(self):
        if not all(isinstance(x, int) and x > 0 for x in [self.seq_len, self.embed_dim, self.num_heads_stacking, self.target_l_new, self.d_new, self.num_heads_latent]):
            raise ValueError("All LMAConfigRL numeric inputs must be positive integers.")
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
        # Corrected: Positional encoding should be registered as a buffer if it's not learnable
        # and needs to be moved to device with the model.
        self.register_buffer("_positional_encoding_tensor", self._create_positional_encoding(lma_config.seq_len, lma_config.embed_dim), persistent=False)
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=bias)
        self.embed_layer_2_act = nn.ReLU()

    def _create_positional_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x shape: (B, L, features_per_step)
        B, L, _ = x.shape
        if L != self.lma_config.seq_len: raise ValueError(f"Input sequence length ({L}) does not match LMAConfigRL.seq_len ({self.lma_config.seq_len}).")
        
        y = self.input_embedding_act(self.input_embedding(x)) # y shape: (B, L, embed_dim)
        
        # Add positional encoding
        # Ensure the positional encoding tensor is on the same device as y
        y = y + self._positional_encoding_tensor[:L, :].to(y.device) # Slicing in case L < pe.seq_len (should not happen with check)
        
        y = self.embedding_dropout(y)
        
        # Reshape for stacking (multi-head attention style)
        d0 = self.lma_config.embed_dim; nh_stack = self.lma_config.num_heads_stacking; dk_stack = d0 // nh_stack
        
        # y shape: (B, L, nh_stack * dk_stack)
        # Reshape to (B, L, nh_stack, dk_stack) then transpose to (B, nh_stack, L, dk_stack)
        # Then reshape to (B, nh_stack * L, dk_stack) - this is effectively concatenating sequences from different "stacking heads"
        y_reshaped_for_stacking = y.view(B, L, nh_stack, dk_stack).transpose(1, 2).contiguous() # (B, nh_stack, L, dk_stack)
        x_stacked_seq = y_reshaped_for_stacking.view(B, nh_stack * L, dk_stack) # (B, L_stacked = nh_stack*L, dk_stack)
                                                                              # This might not be the intended stacking.
                                                                              # The original LMA paper might have a different stacking.
                                                                              # Your original code: head_views = torch.split(y, dk_stack, dim=2)
                                                                              #                    x_stacked_seq = torch.cat(head_views, dim=1)
                                                                              # This results in (B, L * nh_stack, dk_stack) -- which is sequence length times num_heads.
                                                                              # Let's revert to your original stacking logic for consistency.

        head_views = torch.split(y, dk_stack, dim=2) # list of tensors, each (B, L, dk_stack)
        x_stacked_seq_orig = torch.cat(head_views, dim=1) # (B, L * nh_stack, dk_stack) -- L becomes L*nh_stack, embed_dim becomes dk_stack
                                                          # This means total_features = (L * nh_stack) * dk_stack = L * d0. Matches.
                                                          # The LMA config's seq_len and embed_dim refer to the input to this module *before* this split-cat.

        # Flatten the result of split-cat: (B, L * nh_stack * dk_stack) = (B, L * d0)
        x_flat = x_stacked_seq_orig.reshape(B, -1) # (B, L * embed_dim)
        
        # Rechunk according to LMAConfigRL.L_new and C_new
        # L_new * C_new must equal L * embed_dim
        if self.lma_config.L_new * self.lma_config.C_new != L * self.lma_config.embed_dim:
            raise ValueError(f"LMA config L_new*C_new ({self.lma_config.L_new*self.lma_config.C_new}) "
                             f"does not match input total features L*embed_dim ({L*self.lma_config.embed_dim})")

        x_rechunked = x_flat.view(B, self.lma_config.L_new, self.lma_config.C_new)
        
        z = self.embed_layer_2_act(self.embed_layer_2(x_rechunked)) # z shape: (B, L_new, d_new)
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
        if not isinstance(observation_space, GymnasiumBox): # Ensure it's GymnasiumBox
            print(f"Warning: LMAFeaturesExtractor received type {type(observation_space)}. Expected GymnasiumBox.")
        if len(observation_space.shape) != 1: raise ValueError(f"LMAFeaturesExtractor expects a 1D obs space, got {observation_space.shape}.")
        
        self.input_dim_total_flat = observation_space.shape[0] # This is the already-flat dimension for LMA
        self.seq_len_for_initial_transform = seq_len # This is L_orig (num_stacked_frames from env)
        
        if self.input_dim_total_flat % self.seq_len_for_initial_transform != 0:
            raise ValueError(f"Total flat input dim ({self.input_dim_total_flat}) to LMA must be divisible by "
                             f"seq_len_for_initial_transform ({self.seq_len_for_initial_transform}). "
                             f"This implies features_per_step * seq_len_for_initial_transform == input_dim_total_flat.")
        
        self.features_per_step_for_initial_transform = self.input_dim_total_flat // self.seq_len_for_initial_transform

        # This LMAConfigRL is used by LMA_InitialTransform_RL and LMABlock_RL
        # seq_len here is the L that LMA_InitialTransform_RL operates on (e.g. num_stacked_frames)
        # embed_dim here is the d0 that LMA_InitialTransform_RL projects features_per_step into.
        try:
            _lma_config_temp = LMAConfigRL(
                seq_len=self.seq_len_for_initial_transform, # L (e.g. 4 stacked frames)
                embed_dim=embed_dim,                       # d0 (e.g. 64)
                num_heads_stacking=num_heads_stacking,     # e.g. 4
                target_l_new=target_l_new,                 # L' (e.g. 2)
                d_new=d_new,                               # d' (e.g. 32)
                num_heads_latent=num_heads_latent          # e.g. 4
            )
        except ValueError as e:
            raise ValueError(f"Failed to init LMAConfigRL for LMAFeaturesExtractor: {e}") from e
        
        feature_dim_out = _lma_config_temp.L_new * _lma_config_temp.d_new # Final output dim from LMA
        super().__init__(observation_space, features_dim=feature_dim_out)
        
        self.lma_config = _lma_config_temp
        
        self.initial_transform = LMA_InitialTransform_RL(
            features_per_step=self.features_per_step_for_initial_transform, # (e.g. 12 from XPlaneILSFeatureExtractor)
            lma_config=self.lma_config, 
            dropout=dropout, 
            bias=bias
        )
        self.lma_blocks = nn.ModuleList([
            LMABlock_RL(
                lma_config=self.lma_config, # This config has d_new which LMABlock_RL uses
                ff_latent_hidden=ff_latent_hidden, 
                dropout=dropout, 
                bias=bias
            ) for _ in range(num_lma_layers)
        ])
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor: # observations are flat (B, total_flat_input_dim)
        batch_size = observations.shape[0]
        
        # Reshape flat observations to (B, seq_len_for_initial_transform, features_per_step_for_initial_transform)
        # This is (B, L, features_per_step_input_to_LMA_Initial_Transform)
        x_reshaped = observations.view(
            batch_size, 
            self.seq_len_for_initial_transform, 
            self.features_per_step_for_initial_transform
        )
        
        z = self.initial_transform(x_reshaped); # z shape (B, L_new, d_new)
        for block in self.lma_blocks: 
            z = block(z)
        
        features = self.flatten(z); # features shape (B, L_new * d_new)
        return features

# ===============================================
# Stacked LMA Feature Extractor for XPlane ILS (Using the new XPlaneILSFeatureExtractor)
# ===============================================
class StackedLMA_XPlaneILS_FeaturesExtractor(BaseFeaturesExtractor): # Renamed
    """
    A two-stage features extractor for XPlaneILSEnv with stacked frame observations.
    1. Applies XPlaneILSFeatureExtractor to each frame (which is now mostly a pass-through).
    2. Processes the sequence of these features using LMAFeaturesExtractor.
    """
    def __init__(self,
                 observation_space: gymnasium_spaces.Box, # This is the env's observation_space
                 lma_embed_dim_d0: int = 64,       # d0 for LMA_InitialTransform
                 lma_num_heads_stacking: int = 4,  # Number of stacking heads in LMA_InitialTransform
                 lma_target_l_prime: int = None,   # Target L' for LMA (output sequence length from LMA_InitialTransform)
                 lma_d_prime: int = None,          # Target d' for LMA (output embedding dim per L' from LMA_InitialTransform)
                 lma_num_heads_latent: int = 4,    # Number of heads in LatentAttention
                 lma_ff_latent_hidden: int = 128,  # Hidden dim for MLP in LMABlock
                 lma_num_layers: int = 2,          # Number of LMABlocks
                 lma_dropout: float = 0.1,
                 lma_bias: bool = True):

        assert isinstance(observation_space, GymnasiumBox), \
            f"Expected gymnasium.spaces.Box, got {type(observation_space)}"
        assert len(observation_space.shape) == 2, \
            "Observation space for StackedLMA_XPlaneILS_FeaturesExtractor must be 2D (num_stacked_frames, features_per_frame)"

        self.env_num_stacked_frames = observation_space.shape[0] # This is L_orig (e.g., 4)
        self.env_raw_obs_dim_per_frame = observation_space.shape[1] # Should be XPLANE_ILS_RAW_FEATURE_COUNT (12)

        # 1. Per-frame extractor (XPlaneILSFeatureExtractor)
        _single_frame_gym_space = GymnasiumBox( # Use GymnasiumBox directly
            low=observation_space.low[0],
            high=observation_space.high[0],
            shape=(self.env_raw_obs_dim_per_frame,),
            dtype=observation_space.dtype
        )
        _temp_xplane_ils_extractor = XPlaneILSFeatureExtractor(_single_frame_gym_space)
        self.features_per_frame_after_xplane_proc = _temp_xplane_ils_extractor.features_dim # e.g., 12
        del _temp_xplane_ils_extractor

        # 2. LMA Extractor setup
        # Input to LMA is a sequence of features_per_frame_after_xplane_proc
        # LMAFeaturesExtractor expects a flat 1D observation space for its init,
        # representing (L_orig * features_per_frame_after_xplane_proc)
        _lma_total_input_features_flat = self.env_num_stacked_frames * self.features_per_frame_after_xplane_proc
        
        _dummy_lma_flat_obs_space = GymnasiumBox(
            low=-np.inf, high=np.inf, shape=(_lma_total_input_features_flat,), dtype=observation_space.dtype
        )

        # Determine `target_l_new` (L') and `d_new` (d') for LMA based on its input.
        # These are crucial for the LMAConfigRL.
        # If not provided, calculate some defaults.
        _lma_target_l_prime_resolved = lma_target_l_prime
        if _lma_target_l_prime_resolved is None:
            _lma_target_l_prime_resolved = max(1, self.env_num_stacked_frames // 2) # Example: Halve the sequence length

        _lma_d_prime_resolved = lma_d_prime
        if _lma_d_prime_resolved is None:
            # d0 (lma_embed_dim_d0) is the dimension *after* the initial linear projection in LMA_InitialTransform
            # d' is the dimension *after* the second linear projection in LMA_InitialTransform
            _lma_d_prime_resolved = max(1, lma_embed_dim_d0 // 2) # Example: Halve d0

        _temp_lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space,
            embed_dim=lma_embed_dim_d0,               # d0
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_prime_resolved, # L'
            d_new=_lma_d_prime_resolved,              # d'
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=self.env_num_stacked_frames,      # L_orig (number of frames from env)
            dropout=lma_dropout,
            bias=lma_bias
        )
        final_features_dim = _temp_lma_extractor.features_dim # This is L' * d'
        del _temp_lma_extractor

        super().__init__(observation_space, features_dim=final_features_dim)

        # Actual module initializations
        self.xplane_ils_feature_extractor = XPlaneILSFeatureExtractor(_single_frame_gym_space)
        
        self.lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space, # For init, LMA takes flat
            embed_dim=lma_embed_dim_d0,
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_prime_resolved,
            d_new=_lma_d_prime_resolved,
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=self.env_num_stacked_frames, # L_orig for LMA's internal reshaping logic
            dropout=lma_dropout,
            bias=lma_bias
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch_size, env_num_stacked_frames, env_raw_obs_dim_per_frame)
        # e.g., (B, 4, 12)
        batch_size = observations.shape[0]
        
        # Reshape for per-frame processing: (B * L_orig, raw_obs_dim_per_frame)
        obs_reshaped_for_xplane_proc = observations.reshape(
            batch_size * self.env_num_stacked_frames,
            self.env_raw_obs_dim_per_frame
        )
        
        # Process each frame: output (B * L_orig, features_per_frame_after_xplane_proc)
        extracted_features_xplane = self.xplane_ils_feature_extractor(obs_reshaped_for_xplane_proc)
        
        # Reshape for LMA: flat (B, L_orig * features_per_frame_after_xplane_proc)
        processed_obs_for_lma_flat = extracted_features_xplane.reshape(
            batch_size,
            self.env_num_stacked_frames * self.features_per_frame_after_xplane_proc
        )
        
        # LMA processes the flat sequence: output (B, L' * d')
        final_features = self.lma_extractor(processed_obs_for_lma_flat)
        return final_features