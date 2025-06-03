from envs.xplane_ils_env import XPlaneILSEnv
import numpy as np
from xpc.xpc import XPlaneConnect

print("Testing XPlaneConnection ....")
try:
    client = XPlaneConnect()
    client.getPOSI()
    print("XPlaneConnect initialized successfully.")
except Exception as e:
    print("Failed to connect to X-Plane:", e)


env = XPlaneILSEnv()
obs = env.reset()
for _ in range(100):
    action = np.random.uniform(-1, 1, size=4)
    obs, reward, done, _ = env.step(action)
env.close()