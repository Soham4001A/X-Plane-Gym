import gymnasium as gym # if you are using a modern gym/gymnasium
import torch
import argparse
from os import path

# Make sure these imports are correct based on your project structure
from envs.xplane_ils_env import XPlaneILSEnv # Our new environment
from features.LMA_features import StackedLMA_XPlaneILS_FeaturesExtractor # Your custom feature extractor

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def main(args_cli):
    if args_cli.cuda_device and torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device.")
    else:
        device = "cpu"
        print("Using CPU device.")
        if args_cli.cuda_device and not torch.cuda.is_available():
            print("CUDA specified but not available, falling back to CPU.")

    # --- Environment Setup ---
    try:
        env = XPlaneILSEnv(dt=args_cli.env_dt) # dt can be an argument
        print("XPlaneILSEnv created successfully.")
    except ConnectionRefusedError:
        print("FATAL: Could not connect to X-Plane. Ensure X-Plane is running with the XPlaneConnect plugin enabled.")
        return
    except Exception as e:
        print(f"FATAL: Error creating XPlaneILSEnv: {e}")
        return

    # --- Feature Extractor Policy Kwargs ---
    feature_extractor_policy_kwargs = dict(
        features_extractor_class=StackedLMA_XPlaneILS_FeaturesExtractor, # UPDATED NAME
        features_extractor_kwargs=dict(
            # Parameters for your StackedLMA_XPlaneILS_FeaturesExtractor
            lma_embed_dim_d0=args_cli.lma_embed_dim_d0, # d0 (e.g., 64)
            lma_num_heads_stacking=args_cli.lma_num_heads_stacking, # (e.g., 4)
            
            # New explicit params for L' and d'
            lma_target_l_prime=args_cli.lma_target_l_prime, # L' (e.g., NUM_STACKED_FRAMES / 2)
            lma_d_prime=args_cli.lma_d_prime,              # d' (e.g., lma_embed_dim_d0 / 2)

            lma_num_heads_latent=args_cli.lma_num_heads_latent, # (e.g., 4)
            lma_ff_latent_hidden=args_cli.lma_ff_latent_hidden, # (e.g., 128)
            lma_num_layers=args_cli.lma_num_layers, # (e.g., 2)          
            lma_dropout=args_cli.lma_dropout, # (e.g., 0.1)
            lma_bias=True # Typically True
        ),
        net_arch=dict(pi=[128, 64], qf=[128, 64]) # Example for actor/critic network sizes after feature extraction
    )
    
    # Log path for TensorBoard
    log_dir = path.join(path.abspath(path.dirname(__file__)), 'logs_xplane_sac')
    model_save_path = path.join(path.abspath(path.dirname(__file__)), 'models_xplane_sac')

    # Callback for saving models
    checkpoint_callback = CheckpointCallback(
        save_freq=args_cli.save_freq,
        save_path=model_save_path,
        name_prefix="xplane_sac_lma_checkpoint"
    )

    model = None
    model_filename = "xplane_sac_lma_final.zip"
    buffer_filename = "xplane_sac_lma_replay_buffer.pkl"

    if args_cli.load_model and path.exists(path.join(model_save_path, model_filename)):
        print(f"Loading existing model from {path.join(model_save_path, model_filename)}")
        model = SAC.load(
            path.join(model_save_path, model_filename),
            env=env, # Important to pass env for further training
            device=device,
            custom_objects={'policy_kwargs': feature_extractor_policy_kwargs} # if policy_kwargs changed or not saved by default
        )
        if args_cli.load_buffer and path.exists(path.join(model_save_path, buffer_filename)):
            print(f"Loading replay buffer from {path.join(model_save_path, buffer_filename)}")
            model.load_replay_buffer(path.join(model_save_path, buffer_filename))
        print(f"Model and buffer loaded. Training on device: {model.device}")
    else:
        print("No existing model found or load_model not specified. Creating new SAC model.")
        # --- SAC Model Setup ---
        # Hyperparameters for SAC (can be tuned)
        model = SAC(
            'MlpPolicy', # SB3 will use this with the custom features_extractor
            env,
            verbose=1,
            policy_kwargs=feature_extractor_policy_kwargs,
            tensorboard_log=log_dir,
            device=device,
            learning_rate=args_cli.learning_rate,
            buffer_size=args_cli.buffer_size, # 1_000_000
            learning_starts=args_cli.learning_starts, # 10000
            batch_size=args_cli.batch_size, # 256
            tau=0.005, # SAC specific
            gamma=args_cli.gamma, # Discount factor
            train_freq=(args_cli.train_freq_num, args_cli.train_freq_unit), # e.g., (1, "step") or (1, "episode")
            gradient_steps=args_cli.gradient_steps, # How many gradient steps to do after each rollout, -1 for as many as rollout steps
            ent_coef='auto', # Entropy regularization coefficient
            seed=args_cli.seed
        )
        print(f"New SAC Model created. Training on device: {model.device}")

    try:
        print(f"Starting training for {args_cli.total_timesteps} timesteps...")
        model.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=checkpoint_callback,
            log_interval=args_cli.log_interval, # Log every N episodes or steps
            reset_num_timesteps=not args_cli.load_model # False if loading model and continuing training
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e_learn:
        print(f"Error during learning: {e_learn}")
        import traceback
        traceback.print_exc()
    finally:
        if model:
            print("Saving final model and replay buffer...")
            model.save(path.join(model_save_path, model_filename))
            model.save_replay_buffer(path.join(model_save_path, buffer_filename))
            print(f"Final Model saved to {model_save_path}/{model_filename}")
            print(f"Replay Buffer saved to {model_save_path}/{buffer_filename}")
    
    env.close()
    print("Training finished and environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC with StackedLMAFeaturesExtractor on XPlaneILSEnv.")
    # Env args
    parser.add_argument("--env_dt", type=float, default=0.2, help="Environment step time interval (agent decision frequency = 1/dt)")
    
    # Training args
    parser.add_argument("--cuda_device", action="store_true", help="Use CUDA if available")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total timesteps to train for")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N episodes for TensorBoard") # SAC logs per episode
    parser.add_argument("--save_freq", type=int, default=50000, help="Save a checkpoint every N training steps")
    parser.add_argument("--load_model", action="store_true", help="Load a pre-existing model to continue training")
    parser.add_argument("--load_buffer", action="store_true", help="Load a pre-existing replay buffer (use with --load_model)")


    # SAC Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for SAC optimizer") # Common SAC LR
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size for SAC") # Smaller for faster iteration initially
    parser.add_argument("--learning_starts", type=int, default=1000, help="How many steps of random actions before starting to learn") # Let buffer fill a bit
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size for SAC updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq_num", type=int, default=1, help="Train the model every N steps or episodes (number part)")
    parser.add_argument("--train_freq_unit", type=str, default="step", choices=["step", "episode"], help="Train the model every N steps or episodes (unit part)")
    parser.add_argument("--gradient_steps", type=int, default=1, help="How many gradient steps to perform per train_freq interval. -1 means as many as `train_freq_num` if unit is `step`")

    # --- LMA Feature Extractor Specific Args ---
    parser.add_argument("--lma_embed_dim_d0", type=int, default=64, help="LMA: Initial embedding dimension (d0) after first linear projection of per-frame features.")
    parser.add_argument("--lma_num_heads_stacking", type=int, default=4, help="LMA: Number of heads for stacking in LMA_InitialTransform.")
    
    parser.add_argument("--lma_target_l_prime", type=int, default=None, help="LMA: Target sequence length (L') after LMA_InitialTransform's reshaping. If None, defaults (e.g., NUM_STACKED_FRAMES / 2).")
    parser.add_argument("--lma_d_prime", type=int, default=None, help="LMA: Target embedding dimension (d') per L' token after LMA_InitialTransform's reshaping. If None, defaults (e.g., lma_embed_dim_d0 / 2).")
    
    parser.add_argument("--lma_num_heads_latent", type=int, default=4, help="LMA: Number of attention heads in LatentAttention blocks.")
    parser.add_argument("--lma_ff_latent_hidden", type=int, default=128, help="LMA: Hidden dimension size for the MLP in LMABlocks.")
    parser.add_argument("--lma_num_layers", type=int, default=2, help="LMA: Number of LMABlock layers.")
    parser.add_argument("--lma_dropout", type=float, default=0.1, help="LMA: Dropout rate used in LMA components.")

    args = parser.parse_args()
    
    # Some basic validation for train_freq
    if args.train_freq_unit == "episode" and args.gradient_steps != -1 :
        print("Warning: gradient_steps is typically -1 or N when train_freq_unit is 'episode', "
              "to train for N gradient steps at the end of the episode. Setting to 1 for now if it was positive.")
        if args.gradient_steps > 0: args.gradient_steps = 1 # Or set to -1 based on desired SAC behavior for episodic updates

    main(args)