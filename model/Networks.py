import torch
from torch import nn
import torch.nn.functional as F
import pennylane as qml

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
def preprocess_quantum_input(x):
    # batch: shape (batch_size, n_features)
    # Replace nan/infs with zeros
    batch = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Identify all-zero rows
    is_all_zero = (batch.abs().sum(dim=1) < 1e-6)
    # Replace all-zero rows with [1, 0, ..., 0]
    for i, zero_row in enumerate(is_all_zero):
        if zero_row:
            batch[i] = 0
            batch[i, 0] = 1.0
    return batch

n_qubits = 8
n_layers = 2
shots = None # Set to None for simulation

@qml.qnode(qml.device('lightning.qubit', wires=n_qubits, shots=shots), interface='torch', diff_method="adjoint")
def quantum_circuit(inputs, weights):
    weights_index = 0
    # Define the quantum circuit using PennyLane
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)


    # Nearest-neighbors entanglement layer
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class unet(nn.Module):
    def __init__(self, n_classes, n_channels=13, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)


        # Quantum circuit
        quantum_input_shape = 2 ** n_qubits
        self.flatten = nn.Flatten()
        # Hardcoded H and W based on input size 512x512 and 4 downsamples (512/16=32)
        H, W = 32, 32
        self.fc1 = nn.Linear((1024 // factor) * H * W, quantum_input_shape)
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, { "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits) })
        self.fc2 = nn.Linear(n_qubits, (1024 // factor) * H * W)
        self.unflatten = nn.Unflatten(1, (1024 // factor, H, W))

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Quantum component
        x_flat = self.flatten(x5)
        x_fc = self.fc1(x_flat)
        x_fc_clean = preprocess_quantum_input(x_fc)
        try:
            x_quantum = self.quantum_layer(x_fc_clean)
        except Exception as e:
            print(f'exception caught: {e}')
            print(f'input vector: {x_fc_clean}')
            raise e
            
        x_fc2 = self.fc2(x_quantum)
        x5 = self.unflatten(x_fc2)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits