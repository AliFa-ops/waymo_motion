"""
Kinematic math module for spatial normalization.
"""
import math

def get_transformation_matrices(ego_state):
    """Calculates translation offsets and rotation sine/cosine."""
    ref_x = ego_state.center_x
    ref_y = ego_state.center_y
    # Rotate by negative heading to align Ego with X-axis
    cos_h = math.cos(-ego_state.heading)
    sin_h = math.sin(-ego_state.heading)
    return ref_x, ref_y, cos_h, sin_h

def apply_transform(x, y, ref_x, ref_y, cos_h, sin_h):
    """Applies translation and rotation to an X/Y coordinate."""
    dx = x - ref_x
    dy = y - ref_y
    norm_x = dx * cos_h - dy * sin_h
    norm_y = dx * sin_h + dy * cos_h
    return norm_x, norm_y