import numpy as np
import glob

def calculate_heading(vx, vy, valid_mask):
    heading = np.arctan2(vy, vx)
    return heading * valid_mask

def label_swerving(headings, valid_mask, variance_threshold=0.5):
    is_swerving = np.zeros(headings.shape[0], dtype=bool)
    
    for i in range(headings.shape[0]):
        if valid_mask[i, 0] == 1.0:
            valid_headings = headings[i, valid_mask[i] == 1.0]
            if len(valid_headings) > 10:
                heading_variance = np.var(valid_headings)
                if heading_variance > variance_threshold:
                    is_swerving[i] = True
    return is_swerving

def label_lane_change(vy, valid_mask, lateral_vel_threshold=1.0, duration=15):
    is_changing_lane = np.zeros(vy.shape[0], dtype=bool)
    
    for i in range(vy.shape[0]):
        if valid_mask[i, 0] == 1.0:
            valid_vy = np.abs(vy[i, valid_mask[i] == 1.0])
            sustained_lateral_movement = np.sum(valid_vy > lateral_vel_threshold)
            if sustained_lateral_movement >= duration:
                is_changing_lane[i] = True
    return is_changing_lane

if __name__ == "__main__":
    file_paths = glob.glob('../data/processed/*.npy')
    tensor = np.load(file_paths[0])
    
    vx, vy, mask = tensor[:, :, 2], tensor[:, :, 3], tensor[:, :, 5]
    headings = calculate_heading(vx, vy, mask)
    
    swerving_labels = label_swerving(headings, mask)
    lane_change_labels = label_lane_change(vy, mask)
    
    print(f"Agents Swerving: {np.sum(swerving_labels)}")
    print(f"Agents Changing Lanes: {np.sum(lane_change_labels)}")