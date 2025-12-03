import torch                      
from model.Networks import unet   

print("Starting simple test")

# create the model
print("\n1. Creating the model...")
#model predicts 2 things (fire or no fire) input images have 4 channels (satelite data)
model = unet(n_classes=2, n_channels=4)
print("   ✓ Model created!")

# make fake data (2 images, 4 channels, 512x512 pixels)
print("\n2. Creating fake satellite images...")
fake_images = torch.randn(2, 4, 512, 512)
print(f"   ✓ Created fake data with shape: {fake_images.shape}")

# Step 4: Put the fake images through the model
print("\n3. Running images through the model...")
try:
    output = model(fake_images)
    print(f"   ✓ Success! Output shape: {output.shape}")
except Exception as error:
    print(f"   ✗ Error: {error}")
    raise

# Step 5: Done!
print("\n" + "=" * 50)
print("TEST COMPLETE - MODEL WORKS!")
print("=" * 50)







