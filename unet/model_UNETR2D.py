import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config["hidden_dim"], config["mlp_dim"])
        self.dropout1 = nn.Dropout(config["dropout_rate"])
        self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_dim"])
        self.dropout2 = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(config["hidden_dim"])
        self.attention = nn.MultiheadAttention(config["hidden_dim"], config["num_heads"], batch_first=True)
        self.norm2 = nn.LayerNorm(config["hidden_dim"])
        self.mlp = MLP(config)

    def forward(self, x):
        skip_1 = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = skip_1 + x

        skip_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = skip_2 + x

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding='same')

    def forward(self, x):
        x = self.deconv(x)
        return x

class UNETR2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.hidden_dim = config["hidden_dim"]
        self.num_channels = config["num_channels"]
        self.image_size = config["image_size"]
        self.num_patches = config["num_patches"]

        self.input_layer = nn.Linear(config["num_channels"] * config["patch_size"] ** 2, config["hidden_dim"])
        self.position_embedding = nn.Embedding(config["num_patches"], config["hidden_dim"])

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(config) for _ in range(config["num_layers"])
        ])

        self.deconv_layers = nn.ModuleList([
            DeconvBlock(config["hidden_dim"], 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            DeconvBlock(512, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        ])

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1, padding='same')

    def forward(self, x):
        batch_size = x.shape[0]

        # Pad the image so that both dimensions are divisible by the patch size
        _, _, height, width = x.shape
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)

        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size)\
            .unfold(3, self.patch_size, self.patch_size)\
            .contiguous()\
            .view(batch_size, self.num_channels, -1, self.patch_size**2)\
            .permute(0, 2, 1, 3)\
            .contiguous()\
            .view(batch_size, -1, self.num_channels * self.patch_size**2)

        # Patch and position embeddings
        patch_embed = self.input_layer(x)
        positions = torch.arange(0, self.num_patches).unsqueeze(0).to(x.device)
        positions = positions.repeat(batch_size, 1)
        pos_embed = self.position_embedding(positions)
        x = patch_embed + pos_embed


        # Apply transformer encoders
        for encoder in self.transformer_encoders:
            x = encoder(x)

        # Reshape for CNN Decoder
        x = x.permute(0, 2, 1)\
        .contiguous()\
        .view(batch_size, self.hidden_dim, int(self.image_size[0]/self.patch_size), int(self.image_size[1]/self.patch_size))


        # Apply CNN Decoder
        for layer in self.deconv_layers:
            x = layer(x)

        # Apply output layer
        x = self.output_conv(x)

        return x





# # Load the pre-trained VGG-16 model
# self.vgg_backbone = models.vgg16(pretrained=True).features
# # Freeze VGG layers (optional, depending on whether you want to fine-tune them)
#     for param in self.vgg_backbone.parameters():
#         param.requires_grad = False