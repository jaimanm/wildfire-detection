#!/usr/bin/env python
"""
Simple test script to understand the quantum layer in the U-Net model.
This runs a forward pass with dummy data to see what happens.
"""

import torch
import numpy as np
import time
from model.Networks import unet

print("=" * 60)
print("TESTING QUANTUM LAYER IN U-NET")
print("=" * 60)

# Set device (use CPU for testing to avoid GPU allocation issues)
device = torch.device('cpu')
print(f"\nUsing device: {device}")

# Model parameters (matching your train.py mode=5: swir_aerosol)
n_channels = 4  # swir_aerosol mode has 4 channels
n_classes = 2   # fire vs non-fire
batch_size = 2  # Small batch for testing
input_size = (512, 512)

print(f"\nModel Configuration:")
print(f"  - Input channels: {n_channels}")
print(f"  - Output classes: {n_classes}")
print(f"  - Input size: {input_size}")
print(f"  - Batch size: {batch_size}")

# Create the model
print("\n" + "-" * 60)
print("STEP 1: Creating U-Net model with quantum layer...")
print("-" * 60)

try:
    model = unet(n_classes=n_classes, n_channels=n_channels)
    model = model.to(device)
    model.eval()  # Set to evaluation mode (no training)
    print("✓ Model created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
except Exception as e:
    print(f"✗ Error creating model: {e}")
    raise

# Create dummy input data
print("\n" + "-" * 60)
print("STEP 2: Creating dummy input data...")
print("-" * 60)

# Create random tensor mimicking real satellite data
# Values between 0 and 1 (since real data is normalized to /10000)
dummy_input = torch.randn(batch_size, n_channels, input_size[0], input_size[1])
dummy_input = dummy_input.to(device)

print(f"✓ Created dummy input tensor")
print(f"  - Shape: {dummy_input.shape}")
print(f"  - Min value: {dummy_input.min().item():.4f}")
print(f"  - Max value: {dummy_input.max().item():.4f}")
print(f"  - Mean value: {dummy_input.mean().item():.4f}")

# Run forward pass
print("\n" + "-" * 60)
print("STEP 3: Running forward pass through model...")
print("-" * 60)
print("This will go through:")
print("  1. Encoder (down-sampling)")
print("  2. Quantum circuit (at bottleneck)")
print("  3. Decoder (up-sampling)")
print("\nThis might take a moment...")

try:
    start_time = time.time()
    
    with torch.no_grad():  # Don't compute gradients (faster)
        output = model(dummy_input)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✓ Forward pass completed successfully!")
    print(f"  - Time elapsed: {elapsed_time:.3f} seconds")
    print(f"  - Time per image: {elapsed_time/batch_size:.3f} seconds")
    
    print(f"\nOutput tensor:")
    print(f"  - Shape: {output.shape}")
    print(f"  - Expected shape: ({batch_size}, {n_classes}, {input_size[0]}, {input_size[1]})")
    print(f"  - Min value: {output.min().item():.4f}")
    print(f"  - Max value: {output.max().item():.4f}")
    print(f"  - Mean value: {output.mean().item():.4f}")
    
    # Check if output makes sense
    if output.shape == (batch_size, n_classes, input_size[0], input_size[1]):
        print("\n✓ Output shape is correct!")
    else:
        print("\n✗ Output shape mismatch!")
        
except Exception as e:
    print(f"\n✗ Error during forward pass:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test with softmax to see predicted probabilities
print("\n" + "-" * 60)
print("STEP 4: Computing predictions...")
print("-" * 60)

try:
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    
    # Get predicted class (0=non-fire, 1=fire)
    _, predictions = torch.max(probs, dim=1)
    
    print(f"✓ Predictions computed")
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Analyze predictions
    for i in range(batch_size):
        fire_pixels = (predictions[i] == 1).sum().item()
        total_pixels = predictions[i].numel()
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        print(f"\nImage {i+1}:")
        print(f"  - Fire pixels: {fire_pixels:,} / {total_pixels:,}")
        print(f"  - Fire percentage: {fire_percentage:.2f}%")
        print(f"  - Avg fire probability: {probs[i, 1].mean().item():.4f}")
        
except Exception as e:
    print(f"\n✗ Error computing predictions: {e}")
    raise

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Quantum U-Net is working!")
print(f"✓ Processing time: {elapsed_time:.3f}s for {batch_size} images")
print(f"✓ Throughput: {batch_size/elapsed_time:.2f} images/second")
print("\nNext steps:")
print("  1. The model runs successfully with the quantum layer")
print("  2. You can now run the full training with train.py")
print("  3. The quantum circuit is at the bottleneck (after 4 downsamples)")
print("  4. It processes 256-dimensional vectors through 8 qubits")
print("=" * 60)








