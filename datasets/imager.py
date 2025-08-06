"""
These codes are adapted from tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn)
"""

import torch
from torch.utils.data import Dataset

import math


class ImageDataset(Dataset):
    def __init__(self, data, size=100, num_samples=2**18, split='train'):
        super().__init__()
        
        self.data = data
        self.img_wh = (self.data.shape[0], self.data.shape[1])
        self.img_shape = torch.tensor([self.img_wh[0], self.img_wh[1]], dtype=torch.float32)


        print(f"[INFO] image: {self.data.shape}")

        self.num_samples = num_samples
        self.split = split
        self.size = size
        
        if self.split.startswith("test"):
            half_dx =  0.5 / self.img_wh[0]
            half_dy =  0.5 / self.img_wh[1]
            xs = torch.linspace(half_dx, 1-half_dx, self.img_wh[0])
            ys = torch.linspace(half_dy, 1-half_dy, self.img_wh[1])
            xv, yv = torch.meshgrid([xs, ys], indexing="ij")
            xy = torch.stack((xv.flatten(), yv.flatten())).t()
            
            xy_max_num = math.ceil(xy.shape[0] / 1024.0)
            padding_delta = xy_max_num * 1024 - xy.shape[0]
            zeros_padding = torch.zeros((padding_delta, 2))
            self.xs = torch.cat([xy, zeros_padding], dim=0)
            
    def __len__(self):
        return self.size


    def __getitem__(self, _):
        if self.split.startswith('train'):
            xs = torch.rand([self.num_samples, 2], dtype=torch.float32)

            assert torch.sum(xs < 0) == 0, "The coordinates for input image should be non-negative."

            with torch.no_grad():
                scaled_xs = xs * self.img_shape
                indices = scaled_xs.long()
                lerp_weights = scaled_xs - indices.float()

                x0 = indices[:, 0].clamp(min=0, max=self.img_wh[0]-1).long()
                y0 = indices[:, 1].clamp(min=0, max=self.img_wh[1]-1).long()
                x1 = (x0 + 1).clamp(min=0, max=self.img_wh[0]-1).long()
                y1 = (y0 + 1).clamp(min=0, max=self.img_wh[1]-1).long()

                rgbs = self.data[x0, y0] * (1.0 - lerp_weights[:, 0:1]) * (1.0 - lerp_weights[:, 1:2]) + \
                       self.data[x0, y1] * (1.0 - lerp_weights[:, 0:1]) * lerp_weights[:, 1:2] + \
                       self.data[x1, y0] * lerp_weights[:, 0:1] * (1.0 - lerp_weights[:, 1:2]) + \
                       self.data[x1, y1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
        else:
            xs = self.xs
            rgbs = self.data

        results = {
            'points': xs,
            'rgbs': rgbs,
        }

        return results
    
    




# # 🔹 第一段：生成测试阶段的归一化坐标网格

# ```python
# if self.split.startswith("test"):
# ```

# 当你是测试阶段时（例如 `split='test'`），就执行以下代码。

# ---

# ### 📍Step 1: 计算归一化网格点（中心对齐）

# ```python
# half_dx =  0.5 / self.img_wh[0]
# half_dy =  0.5 / self.img_wh[1]
# ```

# * `self.img_wh` 是 `(H, W)`，比如 (256, 256)
# * `half_dx`, `half_dy` 是每个像素在归一化坐标中的 **半宽度**
# * 它们用于生成“**像素中心**”位置的坐标，而不是左上角

# ---

# ```python
# xs = torch.linspace(half_dx, 1-half_dx, self.img_wh[0])
# ys = torch.linspace(half_dy, 1-half_dy, self.img_wh[1])
# ```

# * `xs`: 生成 H 个从上往下的 `x` 轴坐标，范围从 `0.5/H` 到 `1 - 0.5/H`
# * `ys`: 同理，生成 W 个 `y` 轴坐标
# * 这些都是 **归一化的坐标值**，单位是比例而不是像素位置

# ---

# ```python
# xv, yv = torch.meshgrid([xs, ys], indexing="ij")
# ```

# * 生成二维网格坐标：

#   * `xv.shape = [H, W]`，每个点的 x 值
#   * `yv.shape = [H, W]`，每个点的 y 值
# * `indexing="ij"` 表示“行列顺序”（image 格式）

# ---

# ```python
# xy = torch.stack((xv.flatten(), yv.flatten())).t()
# ```

# * 把网格展平为 `[H*W, 2]`，每行是一个 `xy` 坐标
# * `xy.shape = [num_pixels, 2]`，用于传入模型

# ---

# ### 📍Step 2: 补齐长度使其是 1024 的整数倍

# ```python
# xy_max_num = math.ceil(xy.shape[0] / 1024.0)
# padding_delta = xy_max_num * 1024 - xy.shape[0]
# zeros_padding = torch.zeros((padding_delta, 2))
# self.xs = torch.cat([xy, zeros_padding], dim=0)
# ```

# * 原因：**tiny-cuda-nn 通常要求输入长度对齐为 1024 的倍数**
# * 所以这里做了：

#   1. 计算补齐后应该有多少行
#   2. 用 `0` 补齐到 `ceil(HW/1024) * 1024` 个点
#   3. 得到最终的 `self.xs`，shape 为 `[N, 2]`（N 是补齐后的点数）

# ---

# # 🔹 第二段：插值计算每个点对应的 RGB 值

# ⚠️ 这段代码在原始上下文中是错误放在 `__init__` 里的，应放在 `__getitem__()` 的 `test` 分支中。我们还是解释这段代码的逻辑：

# ---

# ## 🧮 Step-by-Step 插值解释

# 我们假设你现在要获取所有 `self.xs` 中的点的 RGB 值。以下操作用于从图像中插值出颜色。

# ---

# ```python
# scaled_xs = xs * self.img_shape
# ```

# * 将 `[0,1]` 中的坐标点 `xs` 映射到实际的图像像素空间
# * `img_shape = [H, W]`
# * 例如 `(0.5, 0.5)` 乘上 `[256, 256]` 得到 `(128, 128)` 像素位置

# ---

# ```python
# indices = scaled_xs.long()
# lerp_weights = scaled_xs - indices.float()
# ```

# * `indices`: 像素索引（下取整），比如 `128.7 → 128`
# * `lerp_weights`: 小数部分，用于计算插值权重，比如 `128.7 - 128 = 0.7`

# ---

# ```python
# x0 = indices[:, 0].clamp(min=0, max=self.img_wh[0]-1).long()
# y0 = indices[:, 1].clamp(min=0, max=self.img_wh[1]-1).long()
# x1 = (x0 + 1).clamp(min=0, max=self.img_wh[0]-1).long()
# y1 = (y0 + 1).clamp(min=0, max=self.img_wh[1]-1).long()
# ```

# * 计算 (x0,y0)、(x0,y1)、(x1,y0)、(x1,y1) 四个邻居像素索引
# * `clamp` 用于防止越界，例如当索引等于边界时 `x+1` 会超过图像大小

# ---

# ## 🔁 插值核心公式（双线性插值）

# ```python
# rgbs = self.data[x0, y0] * (1.0 - wx) * (1.0 - wy) + \
#        self.data[x0, y1] * (1.0 - wx) * wy + \
#        self.data[x1, y0] * wx * (1.0 - wy) + \
#        self.data[x1, y1] * wx * wy
# ```

# 我们用以下变量代替表达式更好理解：

# * `wx = lerp_weights[:, 0:1]`（在 x 轴的小数部分）
# * `wy = lerp_weights[:, 1:2]`（在 y 轴的小数部分）

# 于是：

# ```python
# rgbs = self.data[x0, y0] * (1-wx)*(1-wy) +
#        self.data[x0, y1] * (1-wx)*wy +
#        self.data[x1, y0] * wx*(1-wy) +
#        self.data[x1, y1] * wx*wy
# ```

# 每一项就是“角点 \* 插值权重”的组合，总共 4 项。

# ---

# ## ✅ 最终效果

# 你得到了：

# * 所有点 `xs` 的归一化坐标 `[N, 2]`
# * 对应 RGB 颜色 `rgbs` `[N, 3]`（通过插值获得）

# ---

# ## 🎯 总结：图示解释

# ```text
# 像素：
#      y0   y1
# x0   A    B
# x1   C    D

# 权重：
#      (1-wx)(1-wy) → A
#      (1-wx)(wy)   → B
#      (wx)(1-wy)   → C
#      (wx)(wy)     → D

# f(x,y) = A + B + C + D
# ```

# ---
