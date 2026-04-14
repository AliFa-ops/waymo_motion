import torch
import torch.nn as nn
import torch.optim as optim
import glob
import numpy as np

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_steps=80):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_steps * 2) 
        self.output_steps = output_steps

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        predictions = self.fc(last_hidden)
        return predictions.view(-1, self.output_steps, 2) 

def prep_baseline_data(file_paths):
    inputs, targets = [], []
    for file in file_paths[:50]: 
        tensor = np.load(file)
        ego_data = tensor[0] 
        
        past = ego_data[:11, :] 
        future_xy = ego_data[11:, :2] 
        
        inputs.append(past)
        targets.append(future_xy)
        
    return torch.tensor(np.array(inputs), dtype=torch.float32), \
           torch.tensor(np.array(targets), dtype=torch.float32)

if __name__ == "__main__":
    files = glob.glob('../data/processed/*.npy')
    X_train, Y_train = prep_baseline_data(files)
    
    model = TrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Baseline LSTM Training...")
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print("Training Complete. Baseline model is ready.")