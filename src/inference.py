import torch
from src.datasets.nuscenes_loader import NuScenesSequenceDataset
from src.models.e2e_model import E2EControlModel
from src.utils.metrics import compute_metrics

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NuScenesSequenceDataset(n_prev=3, split='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = E2EControlModel().to(device)
    # model.load_state_dict(torch.load("model_path.pth"))
    
    model.eval()
    with torch.no_grad(): # Back-prop 끄기
        for img_seq, state, label in loader:
            img_seq, state, label = img_seq.to(device), state.to(device), label.to(device)
            output = model(img_seq, state)
            
            metrics = compute_metrics(output, label)
            print(f"Inference - Output: {output.item():.4f}, Target: {label.item():.4f}, Metrics: {metrics}")

if __name__ == "__main__":
    evaluate()