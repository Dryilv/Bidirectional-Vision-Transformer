import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import BiViT  # 导入模型结构

def load_model(model_path, device):
    # 定义与训练时相同的模型结构
    model = BiViT(
        img_size=28,       # 输入图像大小为 28x28
        patch_size=4,      # 分块大小为 4x4
        in_channels=1,     # MNIST 图像是单通道的
        embed_dim=256,     # 嵌入维度设置为 256
        num_heads=8,       # 注意力头数设置为 8
        hidden_dim=512,    # 前馈网络的隐藏层维度
        num_layers=6,      # Transformer 编码器层数
        num_classes=10,    # MNIST 有 10 个分类
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    # 定义与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道
        transforms.Resize((28, 28)),                  # 调整图像大小为 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # 打开图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度，形状变为 [1, 1, 28, 28]
    return image

def predict(image_path, model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    print(f"预测的类别是: {predicted.item()}")

if __name__ == '__main__':
    # 直接指定图像路径
    image_path = 'demo.png'
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
    else:
        predict(image_path)
  