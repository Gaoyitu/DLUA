import torch
from guided_diffusion import Diffusion
from classifier import Classifier

# 加载模型和分类器
model_path = r"D:\work\guided-diffusion-main\256x256_diffusion.pt"
classifier_path = r"D:\work\guided-diffusion-main\256x256_classifier.pt"

diffusion_model = Diffusion(model_path)
classifier_model = Classifier(classifier_path)

# 设置生成参数
num_steps = 1000  # 生成步数
target_label = 3  # 目标标签（根据需要修改）

# 生成图像
with torch.no_grad():
    generated_image = diffusion_model.generate(
        classifier=classifier_model,
        target_label=target_label,
        num_steps=num_steps
    )

# 保存生成的图像
generated_image.save("generated_image.png")
print("图像生成完毕，已保存为generated_image.png")
