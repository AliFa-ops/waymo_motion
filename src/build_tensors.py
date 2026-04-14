"""
Main execution script to process Waymo tfrecords into Deep Learning Tensors.
"""
import tensorflow as tf
import numpy as np
import os
import glob
from waymo_open_dataset.protos import scenario_pb2

import config
import kinematics

def process_scenario(data_bytes, file_name):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(data_bytes)
    
    # Establish Ego Reference Frame (Step 10)
    ego_track = scenario.tracks[scenario.sdc_track_index]
    ref_state = ego_track.states[10]
    ref_x, ref_y, cos_h, sin_h = kinematics.get_transformation_matrices(ref_state)

    # Initialize Padded Tensor [64, 91, 6]
    tensor = np.zeros((config.MAX_AGENTS, config.TIME_STEPS, config.FEATURES), dtype=np.float32)
    
    # Sort tracks (Ego always first at index 0)
    valid_tracks = [ego_track] + [t for i, t in enumerate(scenario.tracks) if i != scenario.sdc_track_index]
    valid_tracks = valid_tracks[:config.MAX_AGENTS]
    
    # Populate Matrix
    for agent_idx, track in enumerate(valid_tracks):
        for t_idx, state in enumerate(track.states):
            if t_idx >= config.TIME_STEPS: break
            
            if state.valid:
                # Normalize Position
                n_x, n_y = kinematics.apply_transform(
                    state.center_x, state.center_y, ref_x, ref_y, cos_h, sin_h)
                
                # Normalize Velocity (Translates around 0,0)
                n_vx, n_vy = kinematics.apply_transform(
                    state.velocity_x, state.velocity_y, 0, 0, cos_h, sin_h)
                
                # Assign Features
                tensor[agent_idx, t_idx, 0] = n_x
                tensor[agent_idx, t_idx, 1] = n_y
                tensor[agent_idx, t_idx, 2] = n_vx
                tensor[agent_idx, t_idx, 3] = n_vy
                tensor[agent_idx, t_idx, 4] = float(track.object_type)
                tensor[agent_idx, t_idx, 5] = 1.0 # The Valid Mask
                
    # Save Tensor Array to Disk
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.OUTPUT_DIR, f"{file_name}.npy")
    np.save(save_path, tensor)

if __name__ == "__main__":
    print("=== Waymo V2X Tensor Pipeline Active ===")
    search_pattern = os.path.join(config.INPUT_DIR, '*.tfrecord*')
    tf_files = sorted(glob.glob(search_pattern))
    
    if not tf_files:
        print(f"ERROR: No .tfrecord files found in {config.INPUT_DIR}")
        exit()
        
    for file_path in tf_files:
        print(f"Processing: {os.path.basename(file_path)}")
        dataset = tf.data.TFRecordDataset(file_path)
        for i, data in enumerate(dataset):
            scenario_id = f"{os.path.basename(file_path)}_scene_{i}"
            process_scenario(data.numpy(), scenario_id)
            
    print("\n Pipeline Complete: All tensors exported to /data/processed/")