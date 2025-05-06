import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import json

# 1. Load pre-trained ResNet-50 model for evaluation
model = resnet50()
model.load_state_dict(torch.load('models/resnet50-0676ba61.pth'))
model.eval()  # set model to evaluation mode

# 2. Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize to 256×256 pixels
    transforms.ToTensor(),  # convert PIL image to PyTorch tensor
    transforms.Normalize(
        (0.5, 0.5, 0.5),  # mean for R, G, B channels
        (0.5, 0.5, 0.5)  # std  for R, G, B channels
    )
])

# 3. Load ground-truth labels from text file
with open('datasets/labels.txt', 'r') as f:
    true_labels = [int(line.strip()) for line in f.readlines()]

# 4. (Optional) Load human-readable class names if needed
with open('datasets/imagenet-simple-labels.json', 'r') as f:
    labels = json.load(f)

# 5. Define the attack target label for counting adversarial successes
attack_target_label = 30

# 6. Initialize counters for overall accuracy and targeted attacks
correct = 0
total = len(true_labels)
target_attack_count = 0

# 7. Iterate over all generated/adversarial images
for i in range(total):
    image_path = os.path.join('result', f"{i + 1}.png")
    image = Image.open(image_path)

    # 7.1 Preprocess and add batch dimension
    image = transform(image).unsqueeze(0)

    # 7.2 Run a forward pass through the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # get class with highest logit score

    # 7.3 Print the true vs predicted labels
    print(f"Image {i + 1}: True Label = {true_labels[i]}, Predicted Label = {predicted.item()}")

    # 7.4 Count as correct if the model predicts the true label (adjusted by -1 if zero-indexed)
    if predicted.item() == true_labels[i] - 1:
        correct += 1

    # 7.5 Count as successful targeted attack if model predicts the attack target label
    if predicted.item() == attack_target_label:
        target_attack_count += 1

# 8. Compute and display classification accuracy
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

# 9. Display how many samples were classified as the attack target
print(f"Number of samples predicted as the attack target label ({attack_target_label}): {target_attack_count}")
