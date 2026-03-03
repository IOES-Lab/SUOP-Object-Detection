import torch
import torch.nn as nn

def sample_and_group(x, npoint):
    B, N, C = x.shape
    if npoint > N:
        idx = torch.cat([
            torch.randperm(N)[:N],
            torch.randint(0, N, (npoint-N,))
        ], dim=0)
    else:
        idx = torch.randperm(N)[:npoint]
    idx = idx.to(x.device)
    new_xyz = x[:, idx, :]
    return new_xyz

class PointNet2SA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.mlp(features)

class PointNet2Detection(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, npoint=512):
        super().__init__()
        self.npoint = npoint
        self.sa1 = PointNet2SA(3, 64)
        self.sa2 = PointNet2SA(64, 128)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_bbox = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 6)
        )
        self.fc_cls = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] != 3 and x.shape[1] == 3:
            x = x.transpose(1, 2)
        if x.shape[-1] != 3:
            raise ValueError(f"x shape must end with 3 (got {x.shape})")
        B, N, C = x.shape
        x = sample_and_group(x, self.npoint)
        features = x.transpose(1, 2)
        features = self.sa1(features)
        features = self.sa2(features)
        x = self.global_pool(features).squeeze(-1)
        bbox = self.fc_bbox(x)
        logits = self.fc_cls(x)
        return bbox, logits

if __name__ == "__main__":
    model = PointNet2Detection(num_classes=5)
    pts = torch.randn(8, 2048, 3)      
    bbox, logits = model(pts)
    print(bbox.shape, logits.shape)     
