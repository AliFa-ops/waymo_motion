import numpy as np
import glob
import os
import config

def analyze_kinematics(tensor):
    valid_mask = tensor[:, :, 5]
    
    vx = tensor[:, :, 2]
    vy = tensor[:, :, 3]
    speeds = np.sqrt(vx**2 + vy**2) * valid_mask
    
    ego_x = tensor[0, :, 0]
    ego_y = tensor[0, :, 1]
    
    distances_to_ego = np.zeros_like(speeds)
    for agent_idx in range(1, tensor.shape[0]):
        agent_x = tensor[agent_idx, :, 0]
        agent_y = tensor[agent_idx, :, 1]
        dist = np.sqrt((agent_x - ego_x)**2 + (agent_y - ego_y)**2)
        distances_to_ego[agent_idx, :] = dist * valid_mask[agent_idx, :]
        
    return speeds, distances_to_ego

def categorize_agents(tensor):
    agent_types = tensor[:, 0, 4] 
    valid_agents = tensor[:, 0, 5] == 1.0
    
    vehicles = np.where((agent_types == 1.0) & valid_agents)[0]
    pedestrians = np.where((agent_types == 2.0) & valid_agents)[0]
    
    return vehicles, pedestrians

if __name__ == "__main__":
    search_path = os.path.join(config.OUTPUT_DIR, '*.npy')
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        print(f"Error: No processed tensors found in {config.OUTPUT_DIR}")
        exit()
        
    sample_tensor = np.load(file_paths[0])
    
    speeds, distances = analyze_kinematics(sample_tensor)
    vehicles, peds = categorize_agents(sample_tensor)
    
    print(f"File analyzed: {os.path.basename(file_paths[0])}")
    print(f"Total Vehicles: {len(vehicles)} | Total Pedestrians: {len(peds)}")
    print(f"Ego Max Speed: {np.max(speeds[0]):.2f} m/s")
    if len(peds) > 0:
        min_ped_dist = np.min(distances[peds[0]][distances[peds[0]] > 0])
        print(f"Closest Pedestrian Distance: {min_ped_dist:.2f} meters")