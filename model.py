import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalCardioAI(nn.Module):
    def __init__(self, clinical_input_size, ecg_feature_dim=128, pcg_feature_dim=128, hidden_dim=64, dropout=0.3):
        super(MultiModalCardioAI, self).__init__()

        # --- ECG branch (ResNet18) ---
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()  # remove original fc layer
        self.ecg_encoder = resnet
        self.ecg_fc = nn.Sequential(
            nn.Linear(512, ecg_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- PCG branch (1D CNN) ---
        self.pcg_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),  # fixed output length
            nn.Flatten(),
            nn.Linear(32 * 128, pcg_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Clinical branch ---
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Fusion layers ---
        self.fc_fusion = nn.Sequential(
            nn.Linear(ecg_feature_dim + pcg_feature_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # for binary classification
        )

    def forward(self, ecg_img, pcg_audio, clinical_data):
        # ECG features
        ecg_feat = self.ecg_fc(self.ecg_encoder(ecg_img))

        # PCG features
        pcg_feat = self.pcg_encoder(pcg_audio)

        # Clinical features
        clinical_feat = self.clinical_fc(clinical_data)

        # Concatenate all modalities
        fused = torch.cat((ecg_feat, pcg_feat, clinical_feat), dim=1)

        # Final output
        out = self.fc_fusion(fused)
        return out
