import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    Per-point MLP -> global max-pool (PointNet-style).
    Input:  B x N x 3
    Output: B x LATENT_DIM
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):  # x: [B, N, 3]
        feat = self.mlp(x)             # [B, N, latent_dim]
        global_feat, _ = torch.max(feat, dim=1)  # [B, latent_dim]
        return global_feat

class Conv3DDecoder(nn.Module):
    """
    Latent -> 8x8x8 seed volume -> upsample via ConvTranspose3d to 64^3.
    Output are logits (no sigmoid inside). Use BCEWithLogitsLoss.
    """
    def __init__(self, latent_dim=128, grid_size=64):
        super().__init__()
        assert grid_size % 8 == 0, "GRID_SIZE should be multiple of 8 (e.g., 32, 64)."
        self.seed_size = 8
        self.gs = grid_size

        seed_channels = 64
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.seed_size * self.seed_size * self.seed_size * seed_channels), nn.ReLU(inplace=True)
        )

        # Upsample 8 -> 16 -> 32 -> 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(seed_channels, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),            # 16->32
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),            # 32->64
            nn.GroupNorm(8, 16), nn.ReLU(inplace=True),

            nn.Conv3d(16, 1, kernel_size=1)  # logits output: [B,1,64,64,64]
        )

    def forward(self, z):  # z: [B, latent_dim]
        B = z.shape[0]
        seed = self.fc(z)
        seed = seed.view(B, 64, self.seed_size, self.seed_size, self.seed_size)
        logits = self.deconv(seed)  # [B,1,gs,gs,gs]
        return logits

# Combined model wrapper
class AE(nn.Module):
    def __init__(self, latent_dim=128, grid_size=64):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = Conv3DDecoder(latent_dim, grid_size)

    def forward(self, x_points):  # x_points: [B,N,3]
        z = self.encoder(x_points)
        logits = self.decoder(z)
        return logits.squeeze(1)  # [B, 64, 64, 64] - return logits directly for BCEWithLogitsLoss


if __name__ == "__main__":
    model = AE()
    x_points = torch.randn(2, 4096, 3)
    logits = model(x_points)
    print(f"Logits shape: {logits.shape}")  # [2, 64, 64, 64]
    probs = torch.sigmoid(logits)
    print(f"Probabilities shape: {probs.shape}")  # [2, 64, 64, 64]