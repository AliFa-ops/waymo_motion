"""
PyTorch DataLoader for the Waymo V2X MotionDiffuser Pipeline.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config

class WaymoMotionDataset(Dataset):
    def __init__(self, data_dir):
        """
        Scans the directory for all processed .npy tensor files.
        """
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}. Did you run build_tensors.py?")
            
        print(f"Dataset Initialized: Found {len(self.file_paths)} scenarios.")

    def __len__(self):
        """Returns the total number of scenarios available."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads a single scenario from the hard drive and converts it to a PyTorch Tensor.
        """
        # Load the numpy array from disk
        file_path = self.file_paths[idx]
        numpy_tensor = np.load(file_path)
        
        # Convert to PyTorch FloatTensor
        torch_tensor = torch.tensor(numpy_tensor, dtype=torch.float32)
        
        return torch_tensor

#  Test the Conveyor Belt
if __name__ == "__main__":
    print("Testing PyTorch DataLoader...")
    
    #Initialize Dataset
    dataset = WaymoMotionDataset(config.OUTPUT_DIR)
    
    # Create the DataLoader (The "Conveyor Belt")
    # batch_size=16 feeding the GPU 16 scenarios at a time.
    # num_workers=4 telling the CPU to use 4 cores to load files in the background while the GPU does math.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # if it works!
    for batch in dataloader:
        print("\n✅ Success! Loaded a batch of data.")
        print(f"Batch Shape: {batch.shape}  -> [Batch_Size, Agents, Time_Steps, Features]")
        break # test only one batch