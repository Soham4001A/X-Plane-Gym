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
                    -180 <= raw_pitch_deg <= 180 and 
                    -360 <= raw_roll_deg <= 360 and  
                    0 <= raw_tas_mps <= 600 and 
                    -100 < raw_alt_agl_m < 30000):
                print(f"WARNING _get_obs: Unphysical DREF values detected...")
                self.current_state_raw = default_crashed_state.copy()
                    #self.current_state_raw = default_crashed_state.copy()
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
            # Return the crashed state observation values, not just zeros
            s_raw = self.current_state_raw
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
                np.clip(s_raw.get("lat_dev_m", self.MAX_LATERAL_DEVIATION_M * 2) / self.NORM_MAX_DEV_M, -1.0, 1.0),
                np.clip(s_raw.get("vert_dev_m", self.MAX_VERTICAL_DEVIATION_M * 2) / self.NORM_MAX_DEV_M, -1.0, 1.0),
                np.clip(s_raw.get("dist_to_thresh_horiz_m", self.NORM_MAX_DISTANCE_M) / self.NORM_MAX_DISTANCE_M, 0.0, 1.0)
            ], dtype=np.float32)
            return obs_values

        # If state was marked invalid, current_state_raw now holds extreme values
        # that should trigger termination in step().
        # If DREFs were okay, calculate deviations:
        if not self.current_state_raw.get("is_state_invalid_from_drefs", False):
            # Fixed the bug here - was using self.current_state_raw["roll_deg"] instead of raw_lat
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

        print("Starting reset procedure...")
        
        # STEP 1: Pause simulation immediately
        self.client.pauseSim(True)
        print("Simulation paused")
        time.sleep(0.5)  # Give X-Plane time to pause
        
        # STEP 2: Fix all systems first
        try:
            print("Fixing all aircraft systems...")
            self.client.sendDREF("sim/operation/fix_all_systems", 1)
            time.sleep(0.2)
            self.client.sendDREF("sim/operation/fix_all_systems", 0)
            print("Systems fixed")
        except Exception as e:
            print(f"Error fixing systems: {e}")
        
        # STEP 3: Reset ALL motion states to zero (this is the key fix!)
        print("Zeroing all motion states...")
        try:
            # Zero all angular velocities (this is crucial for flat spin recovery)
            self.client.sendDREF("sim/flightmodel/position/P", 0.0)  # Roll rate
            self.client.sendDREF("sim/flightmodel/position/Q", 0.0)  # Pitch rate  
            self.client.sendDREF("sim/flightmodel/position/R", 0.0)  # Yaw rate
            
            # Zero all linear velocities
            self.client.sendDREF("sim/flightmodel/position/local_vx", 0.0)  # Local velocity X
            self.client.sendDREF("sim/flightmodel/position/local_vy", 0.0)  # Local velocity Y 
            self.client.sendDREF("sim/flightmodel/position/local_vz", 0.0)  # Local velocity Z
            
            # Zero all accelerations
            self.client.sendDREF("sim/flightmodel/position/local_ax", 0.0)  # Local accel X
            self.client.sendDREF("sim/flightmodel/position/local_ay", 0.0)  # Local accel Y
            self.client.sendDREF("sim/flightmodel/position/local_az", 0.0)  # Local accel Z
            
            print("Motion states cleared")
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error zeroing motion states: {e}")
        
        # STEP 4: Set neutral controls before positioning
        print("Setting neutral controls...")
        self.current_controls = [0.0, 0.0, 0.0, 0.0]  # Zero throttle during reset
        ctrl_cmd = [
            self.current_controls[1], self.current_controls[0], self.current_controls[2], self.current_controls[3],
            self.gear_setting, self.flaps_setting
        ]
        self.client.sendCTRL(ctrl_cmd)
        time.sleep(0.2)
        
        print("Setting throttle to maintain airspeed...")
        self.current_controls[3] = 0.3  # Adjust this value as needed
        ctrl_cmd = [
            self.current_controls[1], self.current_controls[0], self.current_controls[2], self.current_controls[3],
            self.gear_setting, self.flaps_setting
        ]
        self.client.sendCTRL(ctrl_cmd)
        
        # STEP 5: Set aircraft configuration
        print("Setting aircraft configuration...")
        self.client.sendDREF("sim/cockpit2/controls/gear_handle_down", self.gear_setting)
        self.client.sendDREF("sim/cockpit2/controls/flap_ratio", self.flaps_setting)
        time.sleep(0.2)
        
        # STEP 6: Position the aircraft
        print("Positioning aircraft...")
        posi_values = [
            self.initial_lat,           # Latitude
            self.initial_lon,           # Longitude  
            self.initial_alt_msl_ft,    # Altitude MSL in feet
            self.initial_pitch_deg,     # Pitch (0.0)
            self.initial_roll_deg,      # Roll (0.0)
            self.initial_heading_deg,   # Heading
            self.initial_yaw_deg        # Yaw (0.0)
        ]
        self.client.sendPOSI(posi_values)
        time.sleep(0.5)
        
        # STEP 7: Inject forward velocity via local_vx/vy/vz
        print("Injecting forward velocity with override_groundspeed …")

        try:
            # 1) Enable the override so X‑Plane lets us write the velocity vector.
            self.client.sendDREF("sim/operation/override/override_groundspeed", 1)

            # 2) Convert desired airspeed & heading into world‑axis velocity.
            # X‑Plane axes: +X east, +Z south, +Y up
            speed_mps = 80.0                           # For example, 80 m/s = 288 km/h
            hdg_rad   = np.radians(self.initial_heading_deg)
            vx =  speed_mps *  np.sin(hdg_rad)         # east component
            vz = -speed_mps *  np.cos(hdg_rad)         # south = +, north = – so negate cos
            vy =  0.0                                  # level flight for now

            self.client.sendDREF("sim/flightmodel/position/local_vx", vx)
            self.client.sendDREF("sim/flightmodel/position/local_vy", vy)
            self.client.sendDREF("sim/flightmodel/position/local_vz", vz)

            # 3) Zero angular rates so there’s no spin at start
            self.client.sendDREF("sim/flightmodel/position/P", 0.0)
            self.client.sendDREF("sim/flightmodel/position/Q", 0.0)
            self.client.sendDREF("sim/flightmodel/position/R", 0.0)

            time.sleep(0.05)   # give the sim one frame

        finally:
            # 4) Hand velocity control back to the flight model
            self.client.sendDREF("sim/operation/override/override_groundspeed", 0)
        
        # STEP 8: Resume simulation FIRST
        print("Resuming simulation...")
        self.client.pauseSim(False)
        time.sleep(0.5)  # Let physics engine start

        # STEP 9: Get initial observation and populate buffer
        print("Getting initial observations...")
        single_obs = self._get_obs()
        # print(f"DEBUG: single_obs shape: {single_obs.shape}")
        # print(f"DEBUG: single_obs content: {single_obs}")
        
        self.obs_buffer.clear()
        for _ in range(self.NUM_STACKED_FRAMES):
            self.obs_buffer.append(np.copy(single_obs))

        # print(f"DEBUG: obs_buffer length after populate: {len(self.obs_buffer)}")
        # print(f"DEBUG: obs_buffer[0] shape: {self.obs_buffer[0].shape if len(self.obs_buffer) > 0 else 'EMPTY'}")

        final_obs = np.array(self.obs_buffer, dtype=np.float32)
        # print(f"DEBUG: final_obs shape: {final_obs.shape}")

        # Reset episode counters
        self.total_reward = 0
        self.current_step = 0
        
        print("Reset complete!")
        
        # Debug info
        if hasattr(self, 'current_state_raw'):
            s = self.current_state_raw
            print(f"Initial state: Roll={s.get('roll_deg', 'N/A'):.1f}°, Pitch={s.get('pitch_deg', 'N/A'):.1f}°, "
                  f"Heading={s.get('heading_deg', 'N/A'):.1f}°")
            print(f"Rates: P={s.get('P_dps', 'N/A'):.1f}°/s, Q={s.get('Q_dps', 'N/A'):.1f}°/s, "
                  f"R={s.get('R_dps', 'N/A'):.1f}°/s")
            print(f"Speed: {s.get('tas_mps', 'N/A'):.1f} m/s, AGL: {s.get('alt_agl_m', 'N/A'):.1f} m")

        return np.array(self.obs_buffer, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Action: [aileron, elevator, rudder, throttle]
        action = np.clip(action, -1, 1) # Ensure actions are within bounds
        
        # Update current_controls based on agent's action
        self.current_controls = [action[0], action[1], action[2], action[3]]

        # Map to XPlane sendCTRL order: elevator, aileron, rudder, throttle, gear, flaps
        ctrl_cmd = [
            self.current_controls[1], self.current_controls[0], self.current_controls[2], self.current_controls[3],
            self.gear_setting, self.flaps_setting
        ]
        self.client.sendCTRL(ctrl_cmd)
        
        # Wait for dt seconds
        time.sleep(self.dt)
        
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
        reward += (1.0 - stacked_obs[-1, 11]) * 0.1

        # 2. Penalize deviations
        reward -= abs(stacked_obs[-1, 9]) * 0.2  # lat_dev_norm
        reward -= abs(stacked_obs[-1, 10]) * 0.2 # vert_dev_norm
        reward -= abs(stacked_obs[-1, 2]) * 0.1  # heading_error_norm
        reward -= abs(stacked_obs[-1, 3]) * 0.1  # speed_error_norm

        # 3. Penalize excessive roll/pitch beyond a comfort zone
        if abs(s["roll_deg"]) > 15:
             reward -= (abs(s["roll_deg"])-15)/self.NORM_MAX_ROLL_DEG * 0.1
        if abs(s["pitch_deg"]) > 5:
             reward -= (abs(s["pitch_deg"])-5)/self.NORM_MAX_PITCH_DEG * 0.1

        # 4. Survival bonus
        reward += 0.05 

        # --- Termination Conditions ---
        # # Check for flat spin or other dangerous states
        # if (abs(s["P_dps"]) > self.MAX_ROLL_RATE_DPS or 
        #     abs(s["Q_dps"]) > self.MAX_PITCH_RATE_DPS or 
        #     abs(s["R_dps"]) > self.MAX_YAW_RATE_DPS):
        #     reward -= 100
        #     terminated = True
        #     info["termination_reason"] = f"excessive_rates (P:{s['P_dps']:.1f}, Q:{s['Q_dps']:.1f}, R:{s['R_dps']:.1f})"
        
        # Crash / too low
        if s["alt_agl_m"] < 300.0 and s["dist_to_thresh_horiz_m"] > 50:
            reward -= 100
            terminated = True
            info["termination_reason"] = "crashed_short"
        
        # Successfully reached near threshold
        if s["dist_to_thresh_horiz_m"] < 50 and s["alt_agl_m"] < 15 and s["alt_agl_m"] > 0:
            # Speed violations
            if s["tas_mps"] < self.MIN_APPROACH_SPEED_MPS:
                reward -= 50; terminated = True; info["termination_reason"] = "too_slow"
            elif s["tas_mps"] > self.MAX_APPROACH_SPEED_MPS:
                reward -= 50; terminated = True; info["termination_reason"] = "too_fast"
            else:
                # Check approach quality
                on_centerline = abs(s["lat_dev_m"]) < 10
                on_glideslope = abs(s["vert_dev_m"]) < 5
                correct_speed = (self.MIN_APPROACH_SPEED_MPS - 5 < s["tas_mps"] < self.TARGET_APPROACH_SPEED_MPS + 5)
                aligned_heading = abs(s["heading_error_deg"]) < 5
                stable_attitude = abs(s["roll_deg"]) < 5 and -5 < s["pitch_deg"] < 5

                if on_centerline and on_glideslope and correct_speed and aligned_heading and stable_attitude:
                    reward += 200
                    terminated = True
                    info["termination_reason"] = "successful_approach_to_threshold"
                else:
                    reason_details = []
                    if not on_centerline: reason_details.append(f"lat_dev={s['lat_dev_m']:.1f}m")
                    if not on_glideslope: reason_details.append(f"vert_dev={s['vert_dev_m']:.1f}m")
                    if not correct_speed: reason_details.append(f"speed={s['tas_mps']:.1f}mps")
                    if not aligned_heading: reason_details.append(f"hdg_err={s['heading_error_deg']:.1f}deg")
                    if not stable_attitude: reason_details.append(f"roll={s['roll_deg']:.1f}deg, pitch={s['pitch_deg']:.1f}deg")
                    
                    info["termination_reason"] = f"unstable_at_threshold ({', '.join(reason_details)})"
                    reward -= 20
                    terminated = True

        # Max episode steps
        truncated = False
        if self.current_step >= 1000:
            truncated = True
            info["termination_reason"] = "max_steps_reached"
            if not terminated: reward -= 10

        self.total_reward += reward
        if terminated or truncated:
            print(f"Episode ended. Reason: {info.get('termination_reason', 'unknown')}, Steps: {self.current_step}, Total Reward: {self.total_reward:.2f}")
            print(f"Final state: Roll={s['roll_deg']:.1f}, Pitch={s['pitch_deg']:.1f}, TAS={s['tas_mps']:.1f}m/s, AGL={s['alt_agl_m']:.1f}m")
            print(f"LatDev={s['lat_dev_m']:.1f}m, VertDev={s['vert_dev_m']:.1f}m, Dist={s['dist_to_thresh_horiz_m']:.1f}m")

        return stacked_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        try:
            self.client.pauseSim(True)
            print("XPlaneILSEnv closed.")
        except Exception as e:
            print(f"Error during XPlaneILSEnv close: {e}")

if __name__ == '__main__':
    # --- Test the environment ---
    print("Testing XPlaneILSEnv...")
    try:
        env = XPlaneILSEnv(dt=0.5)
        print("XPlaneILSEnv initialized.")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        for i_episode in range(2):
            obs, info = env.reset()
            print(f"Episode {i_episode+1} Reset. Initial obs stack shape: {obs.shape}")
            
            done = False
            total_ep_reward = 0
            for t in range(200):
                action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_ep_reward += reward
                
                if (t + 1) % 10 == 0:
                    print(f"Step {t+1}, Action: [{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}], Reward: {reward:.3f}")

                if done:
                    print(f"Episode finished after {t+1} timesteps. Total Reward: {total_ep_reward:.2f}")
                    break
            if not done:
                print(f"Episode reached max test steps (200). Total Reward: {total_ep_reward:.2f}")
        env.close()

    except ConnectionRefusedError:
        print("X-Plane Connect connection refused. Make sure X-Plane is running with the plugin.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")