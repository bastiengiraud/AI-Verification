import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_metadata(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def export_to_yaml(model, meta, path):
    """Exports model weights and agnostic physics metadata."""
    mapping = meta["mapping"]
    physics = meta["physics"]
    
    data = {
        "model_meta": {
            "name": "lp_proxy",
            "pclass": "optimization",
            "ptype": "lp",
            "architecture": "feedforward",
            "activation": "relu",
            "check": "distance",
            "report": "no",
        },
        "verification_spec": {
            "input_bounds": meta["input_bounds"],
            "indices": mapping,
            "constraints": {
                "A": physics["A"],
                "b_static": physics["b"] 
            },
            "objective_c": physics["c"]
        },
        "layers": [],
        
    }

    for module in model:
        if isinstance(module, nn.Linear):
            data["layers"].append({
                "weights": module.weight.detach().numpy().tolist(),
                "biases": module.bias.detach().numpy().tolist()
            })
            
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    # 0. Load Metadata
    meta_path = BASE_DIR / "lp_metadata.yaml"
    if not meta_path.exists():
        print(f"Metadata not found! Run generation script first.")
        return
    meta = load_metadata(meta_path)
    
    # Extract index mapping
    mapping = meta["mapping"]
    in_idx = mapping['input_indices']
    out_idx = mapping['output_indices']
    
    n_inputs = len(in_idx)
    n_outputs = len(out_idx)

    # 1. Load Data
    data_path = BASE_DIR / "lp_data.csv"
    if not data_path.exists():
        print(f"Data not found at {data_path}!")
        return
    df = pd.read_csv(data_path)
    
    # 2. Extract specific columns based on indices
    X_cols = [f'in_{i}' for i in in_idx]
    Y_cols = [f'out_{j}' for j in out_idx]
    
    X = torch.tensor(df[X_cols].values, dtype=torch.float32)
    Y = torch.tensor(df[Y_cols].values, dtype=torch.float32)

    # 3. Define Architecture (Sized to the mapping)
    model = nn.Sequential(
        nn.Linear(n_inputs, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, n_outputs)
    )

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"[*] Training NN Surrogate...")
    print(f"    Input indices: {in_idx} (Size: {n_inputs})")
    print(f"    Output indices: {out_idx} (Size: {n_outputs})")
    
    for epoch in range(1001): 
        model.train()
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0: 
            print(f"    Epoch {epoch:4d} | Loss: {loss.item():.8f}")

    # 5. Export
    config_path = BASE_DIR / "config.yaml"
    export_to_yaml(model, meta, config_path)
    print(f"\n[+] Training complete. Config exported to: {config_path}")

if __name__ == "__main__":
    main()