"""
Core Neural Network Architecture for Waymo Motion Prediction.
Implements the Social Transformer Encoder.
"""
import torch
import torch.nn as nn

class SocialTransformerEncoder(nn.Module):
    def __init__(self, in_features=6, hidden_dim=128, num_layers=2, n_heads=8):
        super(SocialTransformerEncoder, self).__init__()
        
        # Expands 6 raw numbers into a rich 128-dimensional vector
        self.feature_embedding = nn.Linear(in_features, hidden_dim)
        
        # Transformer Attention - Allows the 64 cars to "talk" to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=512, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Input x shape: [Batch, Agents, Time, Features] -> e.g., [16, 64, 91, 6]
        """
        #  Extract the Padding Mask
        # Look at the Valid Mask (index 5) at the very first time step (index 0)
        # If the mask is 0.0, the agent is just padding
        # PyTorch Transformers require 'True' for elements that should be IGNORED
        valid_mask = x[:, :, 0, 5] 
        padding_mask = (valid_mask == 0.0) # Shape: [16, 64]

        # Temporal Pooling & Embedding
        # For this initial social block, averaging agent's history to get their general trajectory
        # x shape: [16, 64, 91, 6] -> average across Time (dim=2) -> [16, 64, 6]
        x_temporal_pooled = x.mean(dim=2) 
        
        # Embed features: [16, 64, 6] -> [16, 64, 128]
        x_emb = self.feature_embedding(x_temporal_pooled) 

        # Social Attention
        # The cars exchange information, but the padding_mask forces them to ignore the fake padded cars!
        # Output shape: [16, 64, 128]
        social_features = self.transformer(x_emb, src_key_padding_mask=padding_mask)

        return social_features

# Test the Neural Network
if __name__ == "__main__":
    print("Initializing Social Transformer Encoder...")
    model = SocialTransformerEncoder()
    
    # Simulate a fake batch from our DataLoader
    dummy_batch = torch.randn(16, 64, 91, 6)
    
    # Simulate our Valid Mask (Make half the agents valid, half padded)
    dummy_batch[:, :32, :, 5] = 1.0     # First 32 cars are real
    dummy_batch[:, 32:, :, 5] = 0.0     # Last 32 cars are padding
    
    # Pass the batch through the brain
    output = model(dummy_batch)
    
    print("Forward Pass Successful!")
    print(f"Input Shape:  {dummy_batch.shape} (Raw Data)")
    print(f"Output Shape: {output.shape} (Rich Social Embeddings)")