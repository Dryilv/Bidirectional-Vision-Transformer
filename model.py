'''
@author
version
datetime:
'''
import torch
import torch.nn as nn

# 图像分块和嵌入模块
class ImagePatcher(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256):
        super(ImagePatcher, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 线性层将每个分块嵌入为特征向量
        self.patch_embed = nn.Linear(patch_size * patch_size * self.in_channels, embed_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size and width == self.img_size, \
            "输入图像大小必须与 img_size 匹配"
        assert channels == self.in_channels, \
            f"输入图像的通道数({channels})必须与 in_channels({self.in_channels})匹配"

        # 分块
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, -1, self.patch_size * self.patch_size * self.in_channels)  # 展平每个分块

        # 嵌入
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        return x

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.position_embedding

# 双向自注意力模块
class BiDirectionalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(BiDirectionalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 正向和反向多头自注意力层
        self.forward_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.backward_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        seq_len, batch_size, embed_dim = x.shape

        # 正向自注意力
        forward_output, _ = self.forward_attention(x, x, x)

        # 反向自注意力（顺序反转）
        reversed_x = torch.flip(x, [0])
        backward_output, _ = self.backward_attention(reversed_x, reversed_x, reversed_x)
        backward_output = torch.flip(backward_output, [0])

        # 合并正向和反向输出
        combined_output = forward_output + backward_output
        return combined_output

# 双向 Transformer 编码器层
class BiViTEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(BiViTEncoderLayer, self).__init__()
        self.bi_attention = BiDirectionalAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.bi_attention(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x

# 双向 Transformer 编码器
class BiViTEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(BiViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            BiViTEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Bi-ViT 模型
class BiViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(BiViT, self).__init__()
        self.patcher = ImagePatcher(img_size, patch_size, in_channels, embed_dim)
        self.position_encoder = PositionalEncoding(self.patcher.num_patches, embed_dim)
        self.encoder = BiViTEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)

        # 分类标记 (CLS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 分块嵌入
        x = self.patcher(x)
        batch_size, num_patches, _ = x.shape

        # 添加分类标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = self.position_encoder(x)

        # 转置以适应 Bi-ViT：[sequence_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)  # 转回 [batch_size, num_patches + 1, embed_dim]

        # 提取 CLS 标记输出并分类
        x = self.mlp_head(x[:, 0])
        return x

# 遮掩任务模块
class MaskedImageModeling(nn.Module):
    def __init__(self, embed_dim, mask_prob=0.15):
        super(MaskedImageModeling, self).__init__()
        self.embed_dim = embed_dim
        self.mask_prob = mask_prob
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 可学习的 MASK 嵌入向量

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # 生成一个随机遮掩矩阵
        mask = torch.rand(batch_size, num_patches - 1) < self.mask_prob
        mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device), mask), dim=1)  # 不遮掩 CLS token

        # 备份被遮掩的分块嵌入
        masked_tokens = x[mask].clone()

        # 扩展 MASK 嵌入向量的维度以匹配目标维度
        mask_token_expanded = self.mask_token.expand(batch_size, num_patches, embed_dim)
        x[mask] = mask_token_expanded[mask]

        return x, masked_tokens, mask

# 完整的 Bi-ViT 模型，包括遮掩任务
class BiViTWithMaskTask(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, hidden_dim, num_layers, num_classes, mask_prob=0.15, dropout=0.1):
        super(BiViTWithMaskTask, self).__init__()
        self.bi_vit = BiViT(img_size, patch_size, in_channels, embed_dim, num_heads, hidden_dim, num_layers, num_classes, dropout)
        self.masked_image_modeling = MaskedImageModeling(embed_dim, mask_prob)
        self.loss_fn = nn.MSELoss()  # 使用 MSE 损失进行回归

    def forward(self, x):
        # 分块嵌入
        x = self.bi_vit.patcher(x)
        batch_size, num_patches, _ = x.shape

        # 添加分类标记
        cls_tokens = self.bi_vit.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = self.bi_vit.position_encoder(x)

        # 应用遮掩任务
        x, masked_tokens, mask = self.masked_image_modeling(x)

        # 转置以适应 Bi-ViT：[sequence_length, batch_size, embed_dim]
        x = x.transpose(0, 1)
        x = self.bi_vit.encoder(x)
        x = x.transpose(0, 1)  # 转回 [batch_size, num_patches + 1, embed_dim]

        # 提取被遮掩分块的预测嵌入
        predicted_tokens = x[mask]

        # 计算损失
        loss = self.loss_fn(predicted_tokens, masked_tokens)
        return loss

# 示例使用
'''
if __name__ == '__main__':
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    num_heads = 12
    hidden_dim = 3072
    num_layers = 12
    num_classes = 1000
    mask_prob = 0.15

    # 创建模型
    model = BiViTWithMaskTask(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        mask_prob=mask_prob
    )

    # 示例输入图像
    x = torch.randn(8, in_channels, img_size, img_size)  # 8 个图像，3 通道，224x224 大小

    # 前向传播计算损失
    loss = model(x)
    print("Loss:", loss.item())
'''
