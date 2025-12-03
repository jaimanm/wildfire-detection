import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Define a PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define a quantum circuit
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # Encode the inputs into quantum states
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Apply variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])  # Connect last to first

            # Measure expectation values
            return [qml.expval(qml.Z(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        # Initialize weights
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.01)

    def forward(self, x):
        # Ensure input is of shape (batch_size, n_qubits)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            output = self.circuit(x[i], self.weights)
            outputs.append(output)
        return torch.stack(outputs)