"""
3D U-Net architecture for soma segmentation.
Based on the original U-Net paper with 3D adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Bugthanks/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SomaUNet3D(nn.Module):
    """
    Enhanced 3D U-Net specifically designed for soma detection.
    Includes additional features for better soma segmentation.
    """
    
    def __init__(self, n_channels=1, n_classes=1, dropout_rate=0.1, use_attention=False):
        super(SomaUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention
        
        # Encoder
        self.enc1 = self._make_encoder_block(n_channels, 32, dropout_rate)
        self.enc2 = self._make_encoder_block(32, 64, dropout_rate)
        self.enc3 = self._make_encoder_block(64, 128, dropout_rate)
        self.enc4 = self._make_encoder_block(128, 256, dropout_rate)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(256, 512)
        
        # Decoder
        self.dec4 = self._make_decoder_block(512 + 256, 256, dropout_rate)
        self.dec3 = self._make_decoder_block(256 + 128, 128, dropout_rate)
        self.dec2 = self._make_decoder_block(128 + 64, 64, dropout_rate)
        self.dec1 = self._make_decoder_block(64 + 32, 32, dropout_rate)
        
        # Output
        self.final_conv = nn.Conv3d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Max pooling
        self.maxpool = nn.MaxPool3d(2)
        
    def _make_encoder_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
    
    def _make_decoder_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
    
    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc1_pool = self.maxpool(enc1_out)
        
        enc2_out = self.enc2(enc1_pool)
        enc2_pool = self.maxpool(enc2_out)
        
        enc3_out = self.enc3(enc2_pool)
        enc3_pool = self.maxpool(enc3_out)
        
        enc4_out = self.enc4(enc3_pool)
        enc4_pool = self.maxpool(enc4_out)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)
        
        # Decoder
        dec4 = self.dec4(torch.cat([bottleneck, enc4_out], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3_out], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2_out], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1_out], dim=1))
        
        # Output
        out = self.final_conv(dec1)
        out = self.sigmoid(out)
        
        return out


def test_unet():
    """Test the UNet3D model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic UNet
    model = UNet3D(n_channels=1, n_classes=1)
    model = model.to(device)
    
    # Create dummy input
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    # Test enhanced soma UNet
    soma_model = SomaUNet3D(n_channels=1, n_classes=1)
    soma_model = soma_model.to(device)
    
    with torch.no_grad():
        output = soma_model(x)
        print(f"Soma UNet output shape: {output.shape}")


if __name__ == '__main__':
    test_unet()