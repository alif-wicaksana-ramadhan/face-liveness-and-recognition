import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from Modules.model import SimpleEncoder
from Modules.functions import triplet_loss
from Modules.dataset import TripletFaceDataset
from facenet_pytorch import InceptionResnetV1


csv_file = os.path.join('dataset', 'recognition', 'Dataset.csv')  # Replace with the path to your CSV file
root_dir = os.path.join('dataset', 'recognition', 'Faces')    # Replace with the path to your image directory
save_dir = 'model'
model_filename = 'encoder_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on : {}".format(device))

transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize the image
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

os.makedirs(save_dir, exist_ok=True)
custom_dataset = TripletFaceDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, device=device)

# Create a data loader for batching and shuffling
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=True)

embeddingModel = InceptionResnetV1(pretrained='vggface2', device=device).eval()
encoderModel = SimpleEncoder(512, 256, 128).to(device=device)

optimizer = torch.optim.SGD(encoderModel.parameters(), lr=0.01)

num_epochs = 10
verbose_interval = 10

# Iterate through the data loader
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, (anchor_images, positive_images, negative_images) in enumerate(data_loader):
        optimizer.zero_grad()

        # Forward pass to compute embeddings
        anchor_embed = encoderModel(embeddingModel(anchor_images.to(device)))
        positive_embed = encoderModel(embeddingModel(positive_images.to(device)))
        negative_embed = encoderModel(embeddingModel(negative_images.to(device)))

        # Compute triplet loss
        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % verbose_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.4f}")
    
    # Print epoch-level information
    print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {total_loss / len(data_loader):.4f}")

print("Training completed.")
torch.save(encoderModel.state_dict(), os.path.join(save_dir, model_filename))

