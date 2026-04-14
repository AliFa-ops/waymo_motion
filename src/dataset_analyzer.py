"""
Dataset-Wide Behavioral Analyzer
Scans processed Waymo tensors and generates aggregated statistics and visual dashboards.
"""
import numpy as np
import glob
import os
import csv
import matplotlib
matplotlib.use('Agg') # Server-safe plotting
import matplotlib.pyplot as plt

import config
from interaction_tracker import analyze_kinematics, categorize_agents
from behavior_labeler import calculate_heading, label_swerving, label_lane_change

def run_dataset_analysis(max_files=500):
    """
    Scans the dataset and aggregates behavioral metrics.
    max_files: Set to 500 for a quick test, or len(file_paths) for the whole dataset.
    """
    search_path = os.path.join(config.OUTPUT_DIR, '*.npy')
    file_paths = sorted(glob.glob(search_path))
    
    if not file_paths:
        print("Error: No .npy files found.")
        return
        
    files_to_process = file_paths[:max_files]
    print(f"Analyzing {len(files_to_process)} scenarios. Please wait...")

    # --- Aggregation Bins ---
    total_vehicles = 0
    total_pedestrians = 0
    total_swerves = 0
    total_lane_changes = 0
    ego_speeds = []
    close_pedestrian_encounters = 0 # Pedestrians within 5 meters !!

    for i, file in enumerate(files_to_process):
        if (i+1) % 100 == 0:
            print(f"Processing file {i+1}/{len(files_to_process)}...")
            
        tensor = np.load(file)
        valid_mask = tensor[:, :, 5]
        vx, vy = tensor[:, :, 2], tensor[:, :, 3]
        
        speeds, distances = analyze_kinematics(tensor)
        veh, peds = categorize_agents(tensor)
        
        headings = calculate_heading(vx, vy, valid_mask)
        swerving = label_swerving(headings, valid_mask)
        lane_changing = label_lane_change(vy, valid_mask)
        
        total_vehicles += len(veh)
        total_pedestrians += len(peds)
        total_swerves += np.sum(swerving)
        total_lane_changes += np.sum(lane_changing)
        ego_speeds.append(np.max(speeds[0])) # Record the Ego's max speed in this scenario
        
        # Close Pedestrians - Safety Metric !
        if len(peds) > 0:
            for ped_idx in peds:
                min_dist = np.min(distances[ped_idx][distances[ped_idx] > 0])
                if min_dist < 5.0: # Less than 5 meters away
                    close_pedestrian_encounters += 1

    print("\n Generating Visual Dashboard...")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Agent Breakdown
    plt.subplot(1, 3, 1)
    plt.bar(['Vehicles', 'Pedestrians'], [total_vehicles, total_pedestrians], color=['blue', 'green'])
    plt.title("Total Agents Detected")
    plt.ylabel("Count")

    # Plot 2: Ego Speed Distribution
    plt.subplot(1, 3, 2)
    plt.hist(ego_speeds, bins=20, color='purple', alpha=0.7)
    plt.title("Ego-Vehicle Max Speeds")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")

    # Plot 3: Behavioral Events
    plt.subplot(1, 3, 3)
    plt.bar(['Swerves', 'Lane Changes', 'Close Peds (<5m)'], 
            [total_swerves, total_lane_changes, close_pedestrian_encounters], color='red')
    plt.title("Critical Behavioral Events")
    plt.ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, "dataset_analysis_dashboard.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # the CSV Report
    csv_path = os.path.join(config.OUTPUT_DIR, "dataset_report.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Scenarios Analyzed", len(files_to_process)])
        writer.writerow(["Total Vehicles", total_vehicles])
        writer.writerow(["Total Pedestrians", total_pedestrians])
        writer.writerow(["Total Swerving Events", total_swerves])
        writer.writerow(["Total Lane Changes", total_lane_changes])
        writer.writerow(["Close Pedestrian Encounters (<5m)", close_pedestrian_encounters])

    print("Analysis Complete!")
    print(f"Visual Dashboard saved to: {plot_path}")
    print(f"CSV Report saved to: {csv_path}")

if __name__ == "__main__":
    # only 500 files to start with! 
    run_dataset_analysis(max_files=500)

    # Change to run_dataset_analysis(max_files=15000) to run the whole dataset!