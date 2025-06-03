import gymnasium as gym
import numpy as np
from gymnasium import spaces
from xpc.xpc import XPlaneConnect
import time

class XPlaneILSEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define action space: [aileron, elevator, rudder, throttle] ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Define observation space (placeholder values)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Connect to X-Plane
        self.client = XPlaneConnect()
        self.dt = 0.1  # seconds per step

        # Aircraft control channels (XPC order: [ail, elev, rud, throttle])
        self.ctrl = [0.0, 0.0, 0.0, 0.8]  # trimmed level flight default

    def reset(self):
        # Pause simulation
        self.client.pauseSim(True)

        # Set initial state near runway (e.g., 3NM final at 3000 ft AGL)
        # Position: lat, lon, altitude_ft
        init_pos = [37.6188056, -122.3754167, 3000]  # KSFO 28R approximate offset
        self.client.sendPOSI([*init_pos, 0, -3, 0])  # heading = 0°, pitch = -3°, roll = 0°

        # Trimmed flight controls at reset
        self.ctrl = [0.0, 0.0, 0.0, 0.8]  # Neutral aileron/elevator/rudder, 80% throttle
        self.client.sendCTRL(self.ctrl)

        time.sleep(0.5)  # Let sim settle
        self.client.pauseSim(False)

        return self._get_obs()

    def step(self, action):
        # Pause the sim to apply control inputs
        self.client.pauseSim(True)

        # Clip and scale actions to valid control ranges
        action = np.clip(action, -1, 1)
        aileron, elevator, rudder, throttle = action

        self.ctrl = [aileron, elevator, rudder, throttle]
        self.client.sendCTRL(self.ctrl)

        # Advance sim by self.dt
        time.sleep(self.dt)
        self.client.pauseSim(False)
        time.sleep(self.dt)  # Let it simulate for self.dt duration
        self.client.pauseSim(True)

        obs = self._get_obs()

        # Placeholder reward/termination
        reward = 0.0
        done = False

        return obs, reward, done, {}

    def _get_obs(self):
        # Aircraft current position and attitude
        lat = self.client.getDREF("sim/flightmodel/position/latitude")[0]
        lon = self.client.getDREF("sim/flightmodel/position/longitude")[0]
        elev = self.client.getDREF("sim/flightmodel/position/elevation")[0]  # meters MSL

        pitch = self.client.getDREF("sim/flightmodel/position/theta")[0]
        roll = self.client.getDREF("sim/flightmodel/position/phi")[0]
        heading = self.client.getDREF("sim/flightmodel/position/psi")[0]
        tas = self.client.getDREF("sim/flightmodel/position/true_airspeed")[0]
        aoa = self.client.getDREF("sim/flightmodel/position/alpha")[0]

        # Define fixed touchdown point (e.g., KSFO 28R threshold)
        touchdown_lat = 37.615223
        touchdown_lon = -122.389977
        touchdown_elev = 13.0  # meters above MSL

        # Compute position error vector (in lat/lon/elev diff)
        pos_error = np.array([
            lat - touchdown_lat,
            lon - touchdown_lon,
            elev - touchdown_elev
        ])

        # Optionally convert to local meters using flat-earth approximation
        lat_m = (lat - touchdown_lat) * 111320  # meters/deg
        lon_m = (lon - touchdown_lon) * 111320 * np.cos(np.radians(lat))
        alt_m = elev - touchdown_elev

        local_error_meters = np.array([lat_m, lon_m, alt_m])

        obs = np.array([
            lat, lon, elev,
            pitch, roll, heading,
            tas, aoa,
            *local_error_meters  # [x_diff, y_diff, z_diff] in meters
        ], dtype=np.float32)

        return obs

    def render(self, mode="human"):
        pass  # No-op for now

    def close(self):
        self.client.pauseSim(True)