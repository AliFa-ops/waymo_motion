"""
Configuration parameters for the Waymo to MotionDiffuser V2X Pipeline.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Tensor Dimensions
MAX_AGENTS = 64        # Padding threshold
TIME_STEPS = 91        # 9s trajectory at 10Hz + 1 initial frame
FEATURES = 6           # [Norm_X, Norm_Y, Norm_Vel_X, Norm_Vel_Y, Agent_Type, Valid_Mask]
