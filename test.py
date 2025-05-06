import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import json

model = resnet50()
model.load_state_dict(torch.load('models\resnet50-0676ba61.pth'))
model.eval()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

with open('datasets\labels.txt', 'r') as f:
    true_labels = [int(line.strip()) for line in f.readlines()]

with open('datesets\imagenet-simple-labels.json', 'r') as f:
    labels = json.load(f)

attack_target_label = 30

correct = 0
total = len(true_labels)

target_attack_count = 0
for i in range(total):
    image_path = os.path.join(r'result', f"{i + 1}.png")
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f"Image {i + 1}: True Label = {true_labels[i]}, Predicted Label = {predicted.item()}")

    if predicted.item() == true_labels[i] - 1:
        correct += 1

    if predicted.item() == attack_target_label:
        target_attack_count += 1

# 计算准确率
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

print(f"Number of samples predicted as the attack target label ({attack_target_label}): {target_attack_count}")
