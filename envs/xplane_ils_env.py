import gymnasium as gym
import numpy as np
from gymnasium import spaces
from xpc.main import XPlaneConnect
import time
import collections # For deque

class XPlaneILSEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}

    # --- Constants ---
    # Runway KSFO 28R
    RUNWAY_THRESHOLD_LAT = 37.612557
    RUNWAY_THRESHOLD_LON = -122.360427
    RUNWAY_THRESHOLD_ELEV_M = 10.0
    RUNWAY_HEADING_DEG = 281.8

    # Approach Path
    GLIDESLOPE_DEG = 3.0
    INITIAL_ALTITUDE_MSL_FT = 1500.0 # Starting altitude MSL in feet

    # Validated Initial Starting Position (Set 1 from your input)
    INITIAL_START_LAT = 37.538316
    INITIAL_START_LON = -122.287094

    # Aircraft Limits & Targets (Example for a light aircraft like C172)
    MAX_ROLL_DEG = 35.0
    MAX_PITCH_DEG_UP = 25.0
    MAX_PITCH_DEG_DOWN = -20.0 # Allow for descent
    TARGET_APPROACH_SPEED_MPS = 27.0  # Approx 65 KIAS
    MIN_APPROACH_SPEED_MPS = 20.0 # Stall speed with flaps is lower, this is for stable approach
    MAX_APPROACH_SPEED_MPS = 34.0
    MAX_AOA_DEG = 45.0 # Typical stall AOA
    
    MAX_ROLL_RATE_DPS = 60.0
    MAX_PITCH_RATE_DPS = 30.0
    MAX_YAW_RATE_DPS = 30.0

    MAX_LATERAL_DEVIATION_M = 200.0 # Approx full CDI deflection at 1-2NM
    MAX_VERTICAL_DEVIATION_M = 75.0 # Approx full GS deflection

    # Observation Normalization Parameters (some are limits, some are typical scales)
    NORM_MAX_ROLL_DEG = 30.0 # Penalize beyond this, terminate at MAX_ROLL_DEG
    NORM_MAX_PITCH_DEG = 10.0
    NORM_HEADING_ERROR_DEG = 90.0 # Max expected heading error for normalization
    NORM_SPEED_ERROR_MPS = 10.0 # Max expected speed error for normalization
    NORM_AOA_DEG = 10.0 # Max expected AOA for normalization (before stall)
    NORM_MAX_RATES_DPS = 45.0
    NORM_MAX_ALT_AGL_M = 700.0 # Approx initial AGL + buffer
    NORM_MAX_DEV_M = 150.0 # For lateral/vertical deviation normalization

    NORM_MAX_DISTANCE_M = 10.0 * 1852.0 # 10 NM in meters
    NUM_OBS_FEATURES = 12 # Number of features in a single observation frame
    NUM_STACKED_FRAMES = 4 # Number of frames to stack

    def __init__(self, dt=0.2):
        super().__init__()
        gym.utils.EzPickle.__init__(self)
        self.dt = dt
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.NUM_STACKED_FRAMES, self.NUM_OBS_FEATURES),
            dtype=np.float32
        )
        self.client = XPlaneConnect()
        self.gear_setting = 1.0
        self.flaps_setting = 0.25
        self.current_controls = [0.0, 0.0, 0.0, 0.5]
        self.obs_buffer = collections.deque(maxlen=self.NUM_STACKED_FRAMES)

        self.initial_lat = self.INITIAL_START_LAT
        self.initial_lon = self.INITIAL_START_LON
        self.initial_alt_msl_ft = self.INITIAL_ALTITUDE_MSL_FT
        self.initial_heading_deg = self.RUNWAY_HEADING_DEG + 45 # Start offset
        self.initial_pitch_deg = 0.0
        self.initial_roll_deg = 0.0
        self.initial_yaw_deg = 0.0
        
        self.total_reward = 0
        self.current_step = 0

    def _normalize_angle_mpi_pi(self, angle_deg):
        """Normalize angle in degrees to [-180, 180)"""
        angle_deg = angle_deg % 360
        if angle_deg >= 180:
            angle_deg -= 360
        return angle_deg

    def _get_obs(self):
        obs_values_on_error = np.zeros(self.NUM_OBS_FEATURES, dtype=np.float32)
        # Populate self.current_state_raw with values indicative of a crash/error
        # so that termination conditions can catch it based on these values.
        # For example, set AGL to 0, extreme pitch, max deviation, etc.
        default_crashed_state = {
            "roll_deg": 0.0, "pitch_deg": 90.0, "heading_deg": 0.0, # Example: extreme pitch
            "tas_mps": 0.0, "aoa_deg": 90.0, # Example: stall AoA
            "P_dps": self.MAX_ROLL_RATE_DPS * 2, # Example: trigger rate limit
            "Q_dps": self.MAX_PITCH_RATE_DPS * 2,
            "R_dps": self.MAX_YAW_RATE_DPS * 2,
            "alt_agl_m": 0.0, # Crashed
            "lat_dev_m": self.MAX_LATERAL_DEVIATION_M * 2, # Trigger deviation limit
            "vert_dev_m": self.MAX_VERTICAL_DEVIATION_M * 2,
            "dist_to_thresh_horiz_m": self.NORM_MAX_DISTANCE_M, # Far away
            "heading_error_deg": 180.0,
            "is_state_invalid_from_drefs": True # Custom flag
        }

        try:
            raw_lat = self.client.getDREF("sim/flightmodel/position/latitude")[0]
            raw_lon = self.client.getDREF("sim/flightmodel/position/longitude")[0]
            raw_ele_msl_m = self.client.getDREF("sim/flightmodel/position/elevation")[0]
            raw_pitch_deg = self.client.getDREF("sim/flightmodel/position/theta")[0]
            raw_roll_deg = self.client.getDREF("sim/flightmodel/position/phi")[0]
            raw_heading_deg = self.client.getDREF("sim/flightmodel/position/psi")[0]
            raw_tas_mps = self.client.getDREF("sim/flightmodel/position/true_airspeed")[0]
            raw_aoa_deg = self.client.getDREF("sim/flightmodel/position/alpha")[0]
            raw_P_rps = self.client.getDREF("sim/flightmodel/position/P")[0]
            raw_Q_rps = self.client.getDREF("sim/flightmodel/position/Q")[0]
            raw_R_rps = self.client.getDREF("sim/flightmodel/position/R")[0]
            raw_alt_agl_m = self.client.getDREF("sim/flightmodel/position/y_agl")[0]
            
            raw_P_dps = np.degrees(raw_P_rps)
            raw_Q_dps = np.degrees(raw_Q_rps)
            raw_R_dps = np.degrees(raw_R_rps)

            # --- Basic Sanity Checks ---
            # More tolerant limits here, just to catch completely wild X-Plane values
            if not (-90 <= raw_lat <= 90 and -180 <= raw_lon <= 180 and
                    -180 <= raw_pitch_deg <= 180 and # Allow full flip, but not 1500 deg
                    -360 <= raw_roll_deg <= 360 and  # Allow multiple rolls
                    0 <= raw_tas_mps <= 600 and # Mach ~1.7, very generous
                    -100 < raw_alt_agl_m < 30000): # Generous AGL range
                print(f"WARNING _get_obs: Unphysical DREF values detected. Lat:{raw_lat}, Lon:{raw_lon}, Pitch:{raw_pitch_deg}, Roll:{raw_roll_deg}, TAS:{raw_tas_mps}, AGL:{raw_alt_agl_m}")
                # self.current_state_raw = default_crashed_state.copy()
                # Return a normalized version of the crashed state
                # This part needs to be carefully crafted if used, or rely on termination in step()
                # For now, let the large values from default_crashed_state propagate
                # and be caught by termination conditions in step().
                # To directly influence obs:
                # obs_values_on_error[relevant_indices] = normalized_crashed_values
                # For now, we'll just make sure current_state_raw is set to something that *will* terminate.
            else:
                self.current_state_raw = {
                    "roll_deg": raw_roll_deg, "pitch_deg": raw_pitch_deg, "heading_deg": raw_heading_deg,
                    "tas_mps": raw_tas_mps, "aoa_deg": raw_aoa_deg,
                    "P_dps": raw_P_dps, "Q_dps": raw_Q_dps, "R_dps": raw_R_dps,
                    "alt_agl_m": raw_alt_agl_m,
                    # Deviations will be calculated next based on these potentially good DREFs
                }
                self.current_state_raw["is_state_invalid_from_drefs"] = False


        except Exception as e:
            print(f"Error getting DREFs: {e}. Assuming crashed state.")
            self.current_state_raw = default_crashed_state.copy()
            # obs = obs_values_on_error # Or handle normalization of default_crashed_state
            # return obs

        # If state was marked invalid, current_state_raw now holds extreme values
        # that should trigger termination in step().
        # If DREFs were okay, calculate deviations:
        if not self.current_state_raw.get("is_state_invalid_from_drefs", False):
            y_error_m = (self.current_state_raw["roll_deg"] - self.RUNWAY_THRESHOLD_LAT) * 111320.0 # MISTAKE HERE, should be raw_lat
            x_error_m = (raw_lon - self.RUNWAY_THRESHOLD_LON) * (111320.0 * np.cos(np.radians(raw_lat)))
            
            # Re-fetch lat/lon for calculation if not already in current_state_raw, or use the ones read
            # This part of the logic flow for error handling needs to be clean.

            # SAFER: Recalculate deviations only if DREFs were good
            current_lat = self.client.getDREF("sim/flightmodel/position/latitude")[0] # Re-get or use stored raw_lat
            current_lon = self.client.getDREF("sim/flightmodel/position/longitude")[0] # Re-get or use stored raw_lon
            # This re-getting is not ideal. Better to structure so that calculations use the successfully read DREFs.

            # --- Let's simplify: if DREFs are bad, current_state_raw is already populated with crash values ---
            # --- if DREFs are good, calculate normally and populate current_state_raw ---

            # Re-doing the logic cleanly:
            # 1. Try to get DREFs
            # 2. If DREFs bad -> self.current_state_raw = default_crashed_state
            # 3. If DREFs good -> calculate deviations, populate self.current_state_raw with real values
            # 4. Normalize self.current_state_raw to get obs_values

            # Assuming the earlier try-except for DREFs sets self.current_state_raw correctly on error.
            # If no error, proceed to calculate deviations using the valid raw values.

            y_error_m = (raw_lat - self.RUNWAY_THRESHOLD_LAT) * 111320.0
            x_error_m = (raw_lon - self.RUNWAY_THRESHOLD_LON) * (111320.0 * np.cos(np.radians(raw_lat)))
            rwy_hdg_rad = np.radians(self.RUNWAY_HEADING_DEG)
            along_track_projection = x_error_m * np.sin(rwy_hdg_rad) + y_error_m * np.cos(rwy_hdg_rad)
            dist_to_thresh_horiz_m = max(0.0, -along_track_projection)
            lat_dev_m = x_error_m * np.cos(rwy_hdg_rad) - y_error_m * np.sin(rwy_hdg_rad)
            target_agl_on_gs_m = dist_to_thresh_horiz_m * np.tan(np.radians(self.GLIDESLOPE_DEG))
            vert_dev_m = self.current_state_raw["alt_agl_m"] - target_agl_on_gs_m
            heading_error_deg = self._normalize_angle_mpi_pi(self.current_state_raw["heading_deg"] - self.RUNWAY_HEADING_DEG)

            # Update current_state_raw with calculated deviations
            self.current_state_raw["lat_dev_m"] = lat_dev_m
            self.current_state_raw["vert_dev_m"] = vert_dev_m
            self.current_state_raw["dist_to_thresh_horiz_m"] = dist_to_thresh_horiz_m
            self.current_state_raw["heading_error_deg"] = heading_error_deg
            # (Make sure all keys used by normalization exist)


        # Normalize based on self.current_state_raw
        # This ensures that if DREFs were bad, the "crashed" values from default_crashed_state get normalized
        s_raw = self.current_state_raw # Use this for normalization
        obs_values = np.array([
            np.clip(s_raw["roll_deg"] / self.NORM_MAX_ROLL_DEG, -1.0, 1.0),
            np.clip(s_raw["pitch_deg"] / self.NORM_MAX_PITCH_DEG, -1.0, 1.0),
            np.clip(s_raw["heading_error_deg"] / self.NORM_HEADING_ERROR_DEG, -1.0, 1.0),
            np.clip((s_raw["tas_mps"] - self.TARGET_APPROACH_SPEED_MPS) / self.NORM_SPEED_ERROR_MPS, -1.0, 1.0),
            np.clip(s_raw["aoa_deg"] / self.NORM_AOA_DEG, -1.0, 1.0),
            np.clip(s_raw["P_dps"] / self.NORM_MAX_RATES_DPS, -1.0, 1.0),
            np.clip(s_raw["Q_dps"] / self.NORM_MAX_RATES_DPS, -1.0, 1.0),
            np.clip(s_raw["R_dps"] / self.NORM_MAX_RATES_DPS, -1.0, 1.0),
            np.clip(s_raw["alt_agl_m"] / self.NORM_MAX_ALT_AGL_M, 0.0, 1.0),
            np.clip(s_raw.get("lat_dev_m", self.MAX_LATERAL_DEVIATION_M * 2) / self.NORM_MAX_DEV_M, -1.0, 1.0), # .get for safety if key missing
            np.clip(s_raw.get("vert_dev_m", self.MAX_VERTICAL_DEVIATION_M * 2) / self.NORM_MAX_DEV_M, -1.0, 1.0),
            np.clip(s_raw.get("dist_to_thresh_horiz_m", self.NORM_MAX_DISTANCE_M) / self.NORM_MAX_DISTANCE_M, 0.0, 1.0)
        ], dtype=np.float32)
        
        return obs_values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.pauseSim(True)
        time.sleep(0.001)
        
        self.client.sendDREF("sim/cockpit2/controls/gear_handle_down", self.gear_setting)
        self.client.sendDREF("sim/cockpit2/controls/flap_ratio", self.flaps_setting)

        target_ias_knots = self.TARGET_APPROACH_SPEED_MPS * 1.94384
        self.client.sendDREF("sim/flightmodel/position/true_airspeed", self.TARGET_APPROACH_SPEED_MPS)
        self.client.sendDREF("sim/flightmodel/position/indicated_airspeed", target_ias_knots)
        time.sleep(0.1) # Allow time for X-Plane to process the DREFs
        
        x=self.client.getDREF("sim/flightmodel/position/true_airspeed")
        print(f"DEBUG ------- AIRSPEED {x}")
  
        self.current_controls = [0.0, 0.0, 0.0, 0.45]
        ctrl_cmd = [
            self.current_controls[1], self.current_controls[0], self.current_controls[2], self.current_controls[3],
            self.gear_setting, self.flaps_setting
        ]
        self.client.sendCTRL(ctrl_cmd)
        
        # --- FIX ALL SYSTEMS ---
        # This command tells X-Plane to repair any damage from previous crashes/failures.
        try:
            # print("DEBUG RESET: Attempting to fix all systems...")
            self.client.sendDREF("sim/operation/fix_all_systems", 1)
            time.sleep(0.1) # Give X-Plane a moment to process the command
            self.client.sendDREF("sim/operation/fix_all_systems", 0) # Reset the command trigger
            # print("DEBUG RESET: Fix all systems command sent.")
        except Exception as e:
            print(f"DEBUG RESET: Error sending fix_all_systems DREF: {e}")
        # --- END FIX ALL SYSTEMS ---

        # Use the directly set initial lat, lon, alt, heading, pitch, roll
        posi_values = [
            self.initial_lat,           # Using self.INITIAL_START_LAT
            self.initial_lon,           # Using self.INITIAL_START_LON
            self.initial_alt_msl_ft,    # Using self.INITIAL_ALTITUDE_MSL_FT
            self.initial_pitch_deg,     # 0.0
            self.initial_roll_deg,      # 0.0
            self.initial_heading_deg,    # RUNWAY_HEADING_DEG (281.8)
            self.initial_yaw_deg       # 0.0 (or initial yaw if needed)
        ]
        self.client.sendPOSI(posi_values)


        time.sleep(2)
        self.client.pauseSim(False)
        time.sleep(2) # Allow time for X-Plane to stabilize after reset
        
        single_obs = self._get_obs()
        self.obs_buffer.clear()
        for _ in range(self.NUM_STACKED_FRAMES):
            self.obs_buffer.append(np.copy(single_obs))
        
        self.total_reward = 0
        self.current_step = 0

        return np.array(self.obs_buffer, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Action: [aileron, elevator, rudder, throttle]
        action = np.clip(action, -1, 1) # Ensure actions are within bounds
        
        # Update current_controls based on agent's action
        # Aileron, Elevator, Rudder are typically small adjustments from trim
        # Throttle can be more direct.
        # For now, direct mapping:
        self.current_controls = [action[0], action[1], action[2], action[3]]

        # Map to XPlane sendCTRL order: elevator, aileron, rudder, throttle, gear, flaps
        ctrl_cmd = [
            self.current_controls[1], self.current_controls[0], self.current_controls[2], self.current_controls[3],
            self.gear_setting, self.flaps_setting
        ]
        self.client.sendCTRL(ctrl_cmd)
        
        # Wait for dt seconds
        #time.sleep(self.dt)
        
        # Get new observation
        single_obs = self._get_obs()
        self.obs_buffer.append(single_obs)
        stacked_obs = np.array(self.obs_buffer, dtype=np.float32)

        # Calculate reward and check for termination
        reward = 0.0
        terminated = False
        info = {}
        
        s = self.current_state_raw # shorthand for raw state values

        # --- Reward Shaping ---
        # 1. Progress towards runway (closer is better)
        #    Using normalized distance for reward scaling
        reward += (1.0 - stacked_obs[-1, 11]) * 0.1 # obs[-1,11] is normalized dist_to_thresh
                                                    # smaller dist -> larger (1-dist_norm) -> more reward

        # 2. Penalize deviations (the smaller the absolute deviation, the smaller the penalty)
        reward -= abs(stacked_obs[-1, 9]) * 0.2  # lat_dev_norm
        reward -= abs(stacked_obs[-1, 10]) * 0.2 # vert_dev_norm
        reward -= abs(stacked_obs[-1, 2]) * 0.1  # heading_error_norm
        reward -= abs(stacked_obs[-1, 3]) * 0.1  # speed_error_norm

        # 3. Penalize excessive roll/pitch beyond a comfort zone (e.g. > 15 deg roll, > 5 deg pitch)
        if abs(s["roll_deg"]) > 15:
             reward -= (abs(s["roll_deg"])-15)/self.NORM_MAX_ROLL_DEG * 0.1
        if abs(s["pitch_deg"]) > 5: # beyond typical approach pitch
             reward -= (abs(s["pitch_deg"])-5)/self.NORM_MAX_PITCH_DEG * 0.1


        # 4. Survival bonus (small positive reward for each step not terminated)
        reward += 0.05 

        # --- Termination Conditions ---
        # Off track / unstable
        # if abs(s["roll_deg"]) > self.MAX_ROLL_DEG:
        #     reward -= 50; terminated = True; info["termination_reason"] = "excessive_roll"
        # if s["pitch_deg"] > self.MAX_PITCH_DEG_UP or s["pitch_deg"] < self.MAX_PITCH_DEG_DOWN:
        #     reward -= 50; terminated = True; info["termination_reason"] = "excessive_pitch"
        # if abs(s["P_dps"]) > self.MAX_ROLL_RATE_DPS or \
        # abs(s["Q_dps"]) > self.MAX_PITCH_RATE_DPS or \
        # abs(s["R_dps"]) > self.MAX_YAW_RATE_DPS:
        #     reward -= 50; terminated = True; info["termination_reason"] = "excessive_rates"
        
        # Stall
        # if s["aoa_deg"] > self.MAX_AOA_DEG: # Stall
        #      reward -= 100; terminated = True; info["termination_reason"] = "stall_aoa"

        # Deviation limits -- OBE -- We do not use glideslope anymore
        # if abs(s["lat_dev_m"]) > self.MAX_LATERAL_DEVIATION_M:
        #     reward -= 50; terminated = True; info["termination_reason"] = "too_far_laterally"
        # if abs(s["vert_dev_m"]) > self.MAX_VERTICAL_DEVIATION_M:
        #     reward -= 50; terminated = True; info["termination_reason"] = "too_far_vertically"

        # Crash / too low
        # Ground proximity (too low before threshold, or excessive sink rate near ground)
        if s["alt_agl_m"] < 300.0 and s["dist_to_thresh_horiz_m"] > 50: # Crashed before runway
            reward -= 100; terminated = True; info["termination_reason"] = "crashed_short"
        
        # Successfully reached near threshold (placeholder for actual landing success)
        if s["dist_to_thresh_horiz_m"] < 50 and \
           s["alt_agl_m"] < 15 and \
           s["alt_agl_m"] > 0: # Near touchdown point, above ground
               
            # Speed violations
            if s["tas_mps"] < self.MIN_APPROACH_SPEED_MPS:
                reward -= 50; terminated = True; info["termination_reason"] = "too_slow"
            if s["tas_mps"] > self.MAX_APPROACH_SPEED_MPS:
                reward -= 50; terminated = True; info["termination_reason"] = "too_fast"

            # Inner conditions: is the aircraft stable and aligned?
            on_centerline = abs(s["lat_dev_m"]) < 10     # Within 10m laterally
            on_glideslope = abs(s["vert_dev_m"]) < 5      # Within 5m vertically (tight for threshold)
            
            correct_speed = (s["tas_mps"] < (self.TARGET_APPROACH_SPEED_MPS + 5)) and \
                            (s["tas_mps"] > (self.MIN_APPROACH_SPEED_MPS - 5)) # Speed within range

            # NEW: Check for heading alignment
            # s["heading_error_deg"] is (current_heading - runway_heading), normalized to [-180, 180]
            aligned_heading = abs(s["heading_error_deg"]) < 5 # Within +/- 5 degrees of runway heading

            # NEW: Check for reasonable roll and pitch angles (not excessively banked or pitched up/down)
            stable_attitude = abs(s["roll_deg"]) < 5 and \
                              s["pitch_deg"] > -5 and s["pitch_deg"] < 5 # Example: roll < 5deg, pitch between -5 and +5 deg

            if on_centerline and on_glideslope and correct_speed and aligned_heading and stable_attitude:
                reward += 200 # Bonus for reaching threshold in good state
                terminated = True # End episode on successful approach phase completion
                info["termination_reason"] = "successful_approach_to_threshold"
            else:
                # Log which condition failed for better debugging if it's unstable
                reason_details = []
                if not on_centerline: reason_details.append(f"lat_dev={s['lat_dev_m']:.1f}m")
                if not on_glideslope: reason_details.append(f"vert_dev={s['vert_dev_m']:.1f}m")
                if not correct_speed: reason_details.append(f"speed={s['tas_mps']:.1f}mps")
                if not aligned_heading: reason_details.append(f"hdg_err={s['heading_error_deg']:.1f}deg")
                if not stable_attitude: reason_details.append(f"roll={s['roll_deg']:.1f}deg, pitch={s['pitch_deg']:.1f}deg")
                
                info["termination_reason"] = f"unstable_at_threshold ({', '.join(reason_details)})"
                reward -= 20 # Penalize for being near threshold but unstable
                terminated = True


        # Max episode steps
        truncated = False
        if self.current_step >= 1000: # e.g. 1000 steps * 0.2s/step = 200s
            truncated = True
            info["termination_reason"] = "max_steps_reached"
            if not terminated: reward -=10 # Penalize for not reaching destination

        self.total_reward += reward
        if terminated or truncated:
            print(f"Episode ended. Reason: {info.get('termination_reason', 'unknown')}, Steps: {self.current_step}, Total Reward: {self.total_reward:.2f}")
            print(f"Final state: Roll={s['roll_deg']:.1f}, Pitch={s['pitch_deg']:.1f}, TAS={s['tas_mps']:.1f}m/s, AGL={s['alt_agl_m']:.1f}m")
            print(f"LatDev={s['lat_dev_m']:.1f}m, VertDev={s['vert_dev_m']:.1f}m, Dist={s['dist_to_thresh_horiz_m']:.1f}m")

        return stacked_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass # No-op for now, X-Plane itself is the renderer

    def close(self):
        try:
            self.client.pauseSim(True)
            # Optionally send neutral controls before closing
            # ctrl_cmd = [0, 0, 0, 0, self.gear_setting, self.flaps_setting]
            # self.client.sendCTRL(ctrl_cmd)
            print("XPlaneILSEnv closed.")
        except Exception as e:
            print(f"Error during XPlaneILSEnv close: {e}")

if __name__ == '__main__':
    # --- Test the environment ---
    print("Testing XPlaneILSEnv...")
    try:
        env = XPlaneILSEnv(dt=0.5) # Slower dt for easier observation during test
        print("XPlaneILSEnv initialized.")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        for i_episode in range(2):
            obs, info = env.reset()
            print(f"Episode {i_episode+1} Reset. Initial obs stack shape: {obs.shape}")
            # print("Initial single obs (normalized):", obs[-1])
            
            done = False
            total_ep_reward = 0
            for t in range(200): # Max 200 steps per test episode
                action = env.action_space.sample() # Random actions
                # action = np.array([0,0,0,0.4]) # Constant action test
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_ep_reward += reward
                
                if (t + 1) % 10 == 0:
                    print(f"Step {t+1}, Action: [{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}], Reward: {reward:.3f}")
                    # print(f"  Obs (last frame, norm): {obs[-1]}")
                    # print(f"  Raw: Roll={env.current_state_raw['roll_deg']:.1f}, Pitch={env.current_state_raw['pitch_deg']:.1f}, TAS={env.current_state_raw['tas_mps']:.1f}, AGL={env.current_state_raw['alt_agl_m']:.1f}")
                    # print(f"  Devs: Lat={env.current_state_raw['lat_dev_m']:.1f}, Vert={env.current_state_raw['vert_dev_m']:.1f}, Dist={env.current_state_raw['dist_to_thresh_horiz_m']:.1f}")


                if done:
                    print(f"Episode finished after {t+1} timesteps. Total Reward: {total_ep_reward:.2f}")
                    break
            if not done: # If loop finishes before done
                print(f"Episode reached max test steps (200). Total Reward: {total_ep_reward:.2f}")
        env.close()

    except ConnectionRefusedError:
        print("X-Plane Connect connection refused. Make sure X-Plane is running with the plugin.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()