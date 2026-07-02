# CelebRefHQHR 测试集构造说明

## 一、输出结构

```
D:\datasets\CelebRefHQHR\test\
├── x4/                              # r=4 退化级别
│   └── identity_00001/              # 测试身份 (17个)
│       ├── img_0/                   # 以 0.png 做 GT 的测试用例
│       │   ├── gt.png               # 原始高清图
│       │   ├── degraded.png         # 退化后的输入图
│       │   └── conditioning/        # 参考图（同身份其他图片）
│       │       ├── 1.png
│       │       ├── 2.png
│       │       └── ...              # 共14张 (除 gt 自身)
│       ├── img_1/                   # 以 1.png 做 GT …
│       ├── img_2/
│       └── ...                      # ~15个用例/身份
├── x8/                              # r=8 退化级别 (同上结构)
└── x16/                             # r=16 退化级别 (同上结构)
```

## 二、退化公式

$$I_{degraded} = \{[(I_{gt} \otimes k_\sigma) \downarrow_r + n_\delta]\;\text{JPEG}_q\} \uparrow_r$$

五步对应代码 (`restore_dataset.py:160-166`)：

| 步骤 | 操作 | 代码类 |
|------|------|--------|
| 1 | 各向异性高斯模糊 | `CustomGaussianBlur(41, σ_x, σ_y)` |
| 2 | 下采样到 512/r | `Resize(512//r, BILINEAR)` |
| 3 | 加性高斯噪声 | `GaussianNoise(δ)` → 内部 `/255` |
| 4 | JPEG 压缩 | `JPEGCompress(q)` → OpenCV JPEG |
| 5 | 上采样回 512 | `Resize(512, BILINEAR)` |

## 三、退化参数

论文原文 (page 7): "the test set is generated using a **random combination** of noise, blur, and JPEG compression, along with downsampling by ×4, ×8, or ×16."

当前使用的参数（v1，固定值）：

| 级别 | r (下采样倍率) | blur σ | noise δ | JPEG q | 退化程度 |
|------|:---:|:---:|:---:|:---:|------|
| x4 | 4 | 3.0 | 15.0 | 15 | 轻度 |
| x8 | 8 | 5.0 | 17.0 | 14 | 中度 |
| x16 | 16 | 7.0 | 19.0 | 12 | 重度 |

> **注意**：与论文严格对齐的做法应该是 blur/noise/JPEG 参数**随机采样**而非固定值（方案见第五节）。

## 四、数据来源

| 项目 | 值 |
|------|-----|
| 源数据 | `D:\datasets\CelebHQRefForRelease` |
| 总身份数 | 1,005 |
| 训练身份 | 988 个 (排除下面17个) |
| 测试身份 | 17 个 |
| 每身份参考图 | 10~15 张（数量不等） |
| 总测试用例/级别 | 236 张 |

测试身份列表（来自 `gradio_demo.py` `data_dict`）：

```
00001 (Brie Larson)       00027 (Martin Freeman)    00049 (Forest Whitaker)
00052 (Taraji P. Henson)  00062 (Rachel McAdams)    00082 (Chris Pine)
00116 (Gwyneth Paltrow)   00168 (Lil Wayne)         00224 (Blake Lively)
00232 (Angelina Jolie)    00291 (Jake Gyllenhaal)   00435 (Jason Momoa)
00479 (Bradley Cooper)    00621 (George Clooney)     00737 (Michael B. Jordan)
00749 (Mike Tyson)        00757 (Natalie Portman)
```

## 五、改进方案（对标论文）

论文要求测试退化参数**随机采样**，而非固定值。改进脚本需：

1. 对每个测试用例，从训练范围随机采样 blur/noise/JPEG
2. 固定随机种子保证可复现
3. 三者共享同一组退化参数（训练时 r 也是随机的，但测试时 r 固定在指定级别）

```python
# 退化参数采样范围（与训练代码一致）
blur_sigma = np.random.RandomState(seed).uniform(0.1, 12)       # σ ∈ [0.1, 12]
noise = np.random.RandomState(seed).uniform(10, 20)             # δ ∈ [10, 20]
jpeg_q = np.random.RandomState(seed).randint(10, 20)            # q ∈ [10, 20)
downsample = FIXED  # 仅此参数固定: 4 / 8 / 16
```

## 六、与训练时的关系

| | 训练时 | 测试时 |
|---|---|---|
| 退化方式 | 在线随机（`__getitem__` 每次不同） | 预生成存入磁盘（固定随机种子） |
| 退化参数 | 全随机 | r 固定，其余随机 |
| 参考图选取 | 随机 1~4 张（每次不同） | 全部 14 张，推理时由代码随机选 4 张 |
| 评估时行为 | — | `RestoreDatasetTest` 读 `degraded.png` + `gt.png`，代码随机取 conditioning |

## 七、如何重建

```bash
# 当前版本 (固定参数)
cd C:\Users\FJ\Desktop\HrRestorePaper
python scripts/build_celebref_hq_hr.py \
    --source D:/datasets/CelebHQRefForRelease \
    --output D:/datasets/CelebRefHQHR

# 如需仅建测试集
python scripts/build_celebref_hq_hr.py --skip-train
```
