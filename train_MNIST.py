import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # 导入 tqdm 以显示进度条
import os  # 用于创建目录
from model import BiViT  # 从 model.py 导入 BiViT 模型

# 定义早停机制
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path='best_model.pth'):
        """
        Args:
            patience (int): 无改善的 epoch 数量，在达到此数量后停止训练。
            verbose (bool): 如果为 True，则会打印每次验证损失的改善信息。
            delta (float): 判定改善的阈值，只有当新的损失小于 (best_loss - delta) 时才认为是改善。
            save_path (str): 保存最佳模型的路径。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'验证损失未改善 ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型'''
        if self.verbose:
            print(f'验证损失减少，保存模型至 {self.save_path}')
        torch.save(model.state_dict(), self.save_path)

# 加载 MNIST 数据集
def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5, device='cuda'):
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='best_model.pth')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        # 在验证集上评估模型
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        # 检查早停条件
        early_stopping(val_loss, model)

        # 如果触发早停，退出训练
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 验证函数
def evaluate_model(model, data_loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# 测试函数
def test_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集上的模型准确率: {accuracy:.2f}%')

# 主函数
if __name__ == '__main__':
    # 超参数设置
    batch_size = 64
    num_epochs = 50  # 增加 epoch 数，以便观察早停机制
    learning_rate = 0.00001
    patience = 5  # 早停的耐心值

    # 数据加载
    train_loader, val_loader, test_loader = get_mnist_data(batch_size)

    # 模型、损失函数和优化器
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和评估
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device=device)
    test_model(model, test_loader, device=device)