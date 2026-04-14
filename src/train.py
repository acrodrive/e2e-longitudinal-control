import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.nuscenes_loader import NuScenesSequenceDataset
from src.models.e2e_model import E2EControlModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = NuScenesSequenceDataset(n_prev=3)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = E2EControlModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):
        for img_seq, state, label in loader:
            img_seq, state, label = img_seq.to(device), state.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(img_seq, state)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()