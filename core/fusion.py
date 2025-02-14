import torch
import torch.nn as nn

class WorkpieceRepresentationFusionEncoder(nn.Module):
    def __init__(
            self,
            cl_dim = 128,
            cs_dim = 512
    ):
        super().__init__()
        self.fc = nn.Linear(cs_dim, cl_dim)

    def forward(self, feature_x, feature_y):
        b = feature_x.size(0)
        feature_y = feature_y.reshape(b, 1, feature_y.shape[1])
        feature_y = self.fc(feature_y)
        return feature_x + feature_y

class ExternalConditionFusionEncoder(nn.Module):
    def __init__(self, seq_length=60 ,emb_dim = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_length = seq_length
        self.fusion_fc = nn.Linear(emb_dim*2, emb_dim)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        x = self.fusion_fc(x)
        return x
