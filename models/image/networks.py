import torch
import torch.nn as nn
import tinycudann as tcnn

from ..siren import ImageDownsampling, SingleBVPNet


class ImageNGP(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 hidden_dim=64,
                 img_channels=3,  # For RGB image
                 clip_output=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.img_channels = int(img_channels)

        self.clip_output = clip_output
        # Position encoding for 2D coordinates (x, y)
        self.encoder = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )
        input_dim = int(self.encoder.n_output_dims)
        print(f"input_dim: {self.encoder.n_output_dims} (type: {type(self.encoder.n_output_dims)})")

        # MLP to predict RGB values
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = img_channels if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2] in range [-1, 1]
        x = (x + 1) / 2  # Normalize to [0, 1]
        x = self.encoder(x)  # Encoded position features
        rgb = self.backbone(x)  # Predicted RGB

        if self.clip_output:
            rgb = torch.sigmoid(rgb)  # Clamp to [0, 1]

        return rgb
    
    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]

        return params
    
class ImageSiren(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 num_layers=3,
                 img_channels=3,
                 sidelength=256,
                 downsample=False,
                 clip_output=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.img_channels = img_channels
        self.clip_output = clip_output

        self.model = SingleBVPNet(
            in_features=2,
            out_features=img_channels,
            type='sine',
            hidden_features=hidden_dim,
            num_hidden_layers=num_layers,
            mode='mlp',
            sidelength=sidelength,
            downsample=downsample,
        )

    def forward(self, coords):
        """
        coords: [B, 2] in range [-1, 1]
        Returns: RGB [B, 3] in [0, 1] if clip_output is True
        """
        output_dict = self.model({'coords': coords})
        rgb = output_dict['model_out']

        if self.clip_output:
            rgb = torch.sigmoid(rgb)

        return rgb

    @torch.no_grad()
    def get_params(self, LR_schedulers):
        return [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]
