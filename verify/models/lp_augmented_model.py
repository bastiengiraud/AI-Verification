import torch

class LPAugmentedModel(torch.nn.Module):
    def __init__(self, base_model, A, b, in_idx, out_idx):
        super().__init__()
        self.base_model = base_model
        
        total_dim = A.shape[1]  
        num_in = len(in_idx)    
        num_out = len(out_idx)  
        
        # P_in maps NN inputs to their positions
        P_in = torch.zeros((total_dim, num_in))
        for i, idx in enumerate(in_idx):
            P_in[idx, i] = 1.0
            
        # P_out maps NN outputs to their positions
        P_out = torch.zeros((total_dim, num_out))
        for i, idx in enumerate(out_idx):
            P_out[idx, i] = 1.0

        self.register_buffer("P_in", P_in)
        self.register_buffer("P_out", P_out)
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x):
        # x is [batch, 2]
        y = self.base_model(x) # y is [batch, 4]
        
        # Build x_full (batch, 6) using MatMul to stay CROWN-compatible
        # (batch, 2) @ (2, 6) + (batch, 4) @ (4, 6)
        x_full = torch.matmul(x, self.P_in.T) + torch.matmul(y, self.P_out.T)
        
        # residuals: [batch, 5] (one for each constraint row)
        residuals = torch.matmul(x_full, self.A.T) - self.b

        return residuals