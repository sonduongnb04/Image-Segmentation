# Dự án Phân đoạn Ảnh Giao Thông sử dụng Deep Learning

## 📋 Tổng quan

Dự án phân đoạn ảnh giao thông (Traffic Image Segmentation) sử dụng các mô hình Deep Learning tiên tiến để phân đoạn các đối tượng trong ảnh giao thông thành 12 lớp khác nhau. Dự án được xây dựng với hai phần chính:
- **ImageSegmentation**: Xử lý dữ liệu, huấn luyện và đánh giá mô hình
- **Webb**: Giao diện web hoàn chỉnh với Back-End API và Front-End người dùng

## 🎯 Mục tiêu

- Phân đoạn chính xác các đối tượng giao thông trong ảnh
- Hỗ trợ 12 lớp đối tượng: Bầu trời, Tòa nhà, Cột, Đường, Vỉa hè, Cây, Biển báo, Hàng rào, Xe hơi, Người đi bộ, Người đi xe đạp, Không xác định
- Cung cấp giao diện web thân thiện để sử dụng mô hình
- Đạt hiệu suất cao với các metrics như IoU, Dice Score, Pixel Accuracy

## 🏗️ Kiến trúc Hệ thống

```
Segmentation/
├── ImageSegmentation/          # Phần xử lý dữ liệu và huấn luyện mô hình
│   ├── data.py                 # Xử lý và chuẩn bị dữ liệu CamVid
│   ├── model.py                # Mô hình UNet và training loop
│   ├── unetPlus.py             # Mô hình UNet++ với encoder pre-trained
│   ├── demo.py                 # Script demo và test mô hình
│   ├── processed.py            # Tiền xử lý dữ liệu
│   ├── CamVid/                 # Dữ liệu gốc CamVid dataset
│   ├── Camvid_processed/       # Dữ liệu đã được xử lý
│   └── *.npy                   # Files dữ liệu đã được chia train/val/test
├── Webb/                       # Ứng dụng web
│   ├── Back-End/               # API Flask server
│   │   ├── app.py              # Flask application
│   │   ├── model.py            # Định nghĩa mô hình UNet
│   │   ├── requirements.txt    # Dependencies
│   │   └── best_model.pth      # Mô hình đã huấn luyện
│   └── Front-End/              # Giao diện người dùng
│       ├── index.html          # Trang chủ upload ảnh
│       ├── edit.html           # Trang chỉnh sửa và xem kết quả
│       ├── script.js           # Logic JavaScript
│       └── *.css               # Styling
└── README.md                   # Tài liệu này
```
    best_model_py: https://drive.google.com/file/d/1J1II1ZjSKDkbzBoYZlhArNtBMpCarGN1/view?usp=drive_link (do file quá nặng không tải lên github được)
## 🧠 Mô hình Deep Learning

### 1. UNet (model.py)

**Kiến trúc:**
- **Encoder (Downsampling)**: 4 tầng convolution blocks với MaxPooling
- **Bottleneck**: Tầng trung gian xử lý features có độ phân giải thấp nhất
- **Decoder (Upsampling)**: 4 tầng deconvolution với skip connections
- **Skip Connections**: Kết nối trực tiếp từ encoder đến decoder tương ứng

**Đặc điểm kỹ thuật:**
```python
# Cấu hình mô hình
Input channels: 3 (RGB)
Output channels: 12 (12 lớp phân đoạn)
Input size: 256x256
Architecture: U-Net với BatchNorm và ReLU activation

# Conv Block structure
Conv2d(3x3) → BatchNorm2d → ReLU → Conv2d(3x3) → BatchNorm2d → ReLU
```

**Ưu điểm:**
- Skip connections giữ nguyên thông tin chi tiết từ tầng encoder
- Hiệu quả cho các nhiệm vụ segmentation y tế và giao thông
- Kiến trúc đơn giản, dễ huấn luyện

### 2. UNet++ (unetPlus.py)

**Kiến trúc:**
- Sử dụng thư viện `segmentation_models_pytorch`
- **Encoder**: ResNet34 pre-trained trên ImageNet
- **Decoder**: Nested skip connections với nhiều tầng kết nối chéo
- **Dense Skip Connections**: Kết nối đậm đặc giữa các tầng

**Đặc điểm kỹ thuật:**
```python
# Cấu hình UNet++
Encoder: "resnet34" with ImageNet pretrained weights
Input channels: 3
Output classes: 12
Architecture: UNet++ with nested skip pathways
```

**Ưu điểm:**
- Dense skip connections cải thiện khả năng học features
- Encoder pre-trained giúp hội tụ nhanh hơn
- Hiệu suất cao hơn UNet truyền thống

## 📊 Dataset và Xử lý Dữ liệu

### CamVid Dataset
- **Tổng số ảnh**: 701 ảnh giao thông đô thị
- **Kích thước**: 960x720 pixels (được resize về 256x256)
- **12 lớp phân đoạn** với color mapping:

| Lớp | Màu RGB | Mô tả |
|-----|---------|-------|
| 0 | (128, 128, 128) | Bầu trời (Sky) |
| 1 | (128, 0, 0) | Tòa nhà (Building) |
| 2 | (192, 192, 128) | Cột (Pole) |
| 3 | (128, 64, 128) | Đường (Road) |
| 4 | (60, 40, 222) | Vỉa hè (Pavement) |
| 5 | (128, 128, 0) | Cây (Tree) |
| 6 | (192, 128, 128) | Biển báo (SignSymbol) |
| 7 | (64, 64, 128) | Hàng rào (Fence) |
| 8 | (64, 0, 128) | Xe hơi (Car) |
| 9 | (64, 64, 0) | Người đi bộ (Pedestrian) |
| 10 | (0, 128, 192) | Người đi xe đạp (Bicyclist) |
| 11 | (0, 0, 0) | Không xác định (Unlabeled) |

### Data Augmentation
```python
# Training augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),          # Lật ngang ngẫu nhiên
    A.RandomBrightnessContrast(p=0.2), # Thay đổi độ sáng/tương phản
    A.RandomCrop(width=224, height=224), # Cắt ngẫu nhiên
    A.Rotate(limit=10, p=0.3),        # Xoay nhẹ
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Validation/Test augmentation
val_transform = A.Compose([
    A.Resize(height=256, width=256),   # Resize cố định
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
```

### Chia dữ liệu
- **Training set**: 70% (~490 ảnh)
- **Validation set**: 15% (~105 ảnh)  
- **Test set**: 15% (~106 ảnh)

## 🔧 Training Process

### Loss Function
**Combined Loss** = 0.7 × CrossEntropyLoss + 0.3 × DiceLoss

```python
def combined_loss(outputs, targets):
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    return 0.7 * ce_loss(outputs, targets) + 0.3 * dice_loss(outputs, targets)
```

**DiceLoss** được thiết kế đặc biệt cho segmentation:
```python
class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
```

### Optimizer và Learning Rate
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
batch_size = 16
epochs = 50-100 (với Early Stopping)
```

### Early Stopping
- **Patience**: 10 epochs
- **Metric**: Dice + IoU score
- **Mode**: Maximize (điểm càng cao càng tốt)

## 📈 Metrics đánh giá

### 1. Pixel Accuracy
```python
def pixel_accuracy(preds, masks):
    correct = (preds == masks).float().sum()
    return correct / masks.numel()
```

### 2. Mean Intersection over Union (mIoU)
```python
def mean_iou(preds, masks, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        union = ((preds == cls) | (masks == cls)).sum().item()
        ious.append(intersection / union if union > 0 else float('nan'))
    return np.nanmean(ious)
```

### 3. Dice Score
```python
def dice_score(preds, masks, num_classes):
    dices = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        total = (preds == cls).sum().item() + (masks == cls).sum().item()
        dices.append(2 * intersection / total if total > 0 else float('nan'))
    return np.nanmean(dices)
```

## 🌐 Ứng dụng Web (Webb)

### Back-End (Flask API)

**Cấu trúc API:**
```python
# Health check endpoint
GET /api/health
Response: {"status": "ok"}

# Image segmentation endpoint  
POST /api/segment
Input: MultipartForm với file ảnh
Output: {
    "success": true,
    "segmented_image": "data:image/png;base64,..."
}
```

**Quy trình xử lý:**
1. **Upload**: Nhận ảnh từ client
2. **Preprocessing**: Resize về 256x256, normalize
3. **Inference**: Chạy mô hình UNet đã train
4. **Postprocessing**: Chuyển prediction thành mask màu
5. **Response**: Trả về ảnh đã phân đoạn dạng base64

**Dependencies:**
```
flask==2.0.1
flask-cors==3.0.10
torch==1.13.1
torchvision==0.14.1
numpy==1.23.4
opencv-python==4.7.0.72
pillow==9.3.0
albumentations==1.3.0
```

### Front-End (HTML/CSS/JavaScript)

**Tính năng:**
- Upload ảnh với drag & drop
- Preview ảnh trước khi xử lý
- Hiển thị kết quả phân đoạn
- Download ảnh kết quả
- Giao diện responsive, hỗ trợ đa ngôn ngữ (VN/EN)

**Tech Stack:**
- **HTML5**: Cấu trúc semantic
- **CSS3**: Styling modern với Flexbox/Grid
- **Vanilla JavaScript**: Logic xử lý không dependencies
- **Font**: Google Fonts (Poppins)

## 🚀 Hướng dẫn Cài đặt

### 1. Clone Repository
```bash
git clone <repository-url>
cd Segmentation
```

### 2. Cài đặt Dependencies

**Cho ImageSegmentation:**
```bash
cd ImageSegmentation
pip install torch torchvision numpy opencv-python albumentations matplotlib tqdm segmentation-models-pytorch
```

**Cho Webb Back-End:**
```bash
cd Webb/Back-End
pip install -r requirements.txt
```

### 3. Chuẩn bị Dữ liệu

**Download CamVid dataset:**
```bash
# Đặt dữ liệu CamVid vào thư mục ImageSegmentation/CamVid/
# Chạy script xử lý dữ liệu
cd ImageSegmentation
python data.py
```

### 4. Huấn luyện Mô hình

**UNet:**
```bash
python model.py
```

**UNet++:**
```bash
python unetPlus.py
```

### 5. Chạy Ứng dụng Web

**Khởi động Back-End:**
```bash
cd Webb/Back-End
python app.py
# Server sẽ chạy tại http://localhost:5000
```

**Mở Front-End:**
```bash
cd Webb/Front-End
# Mở index.html trong trình duyệt hoặc dùng live server
```

## 📊 Kết quả Thực nghiệm

### Hiệu suất Mô hình

| Mô hình | Pixel Accuracy | mIoU | Dice Score | Training Time |
|---------|---------------|------|------------|---------------|
| UNet | ~85-90% | ~0.72-0.78 | ~0.75-0.82 | 2-3h |
| UNet++ | ~88-92% | ~0.75-0.82 | ~0.78-0.85 | 3-4h |

### Benchmark trên từng lớp

| Lớp | IoU | Dice | Độ khó |
|-----|-----|------|--------|
| Road | >0.9 | >0.95 | Dễ |
| Building | >0.8 | >0.85 | Trung bình |
| Sky | >0.85 | >0.9 | Dễ |
| Car | 0.7-0.8 | 0.75-0.85 | Trung bình |
| Pedestrian | 0.5-0.7 | 0.6-0.75 | Khó |
| Bicyclist | 0.4-0.6 | 0.5-0.7 | Rất khó |

## 🔬 Chi tiết Kỹ thuật

### Optimizations

1. **Mixed Precision Training**: Sử dụng AMP để tăng tốc
2. **Data Loading**: Multiprocessing cho DataLoader
3. **Memory Management**: Gradient accumulation cho batch size lớn
4. **Model Checkpointing**: Lưu best model dựa trên validation metrics

### Hardware Requirements

**Minimum:**
- GPU: 4GB VRAM (GTX 1060/RTX 2060)
- RAM: 8GB
- Storage: 5GB free space

**Recommended:**
- GPU: 8GB+ VRAM (RTX 3070/4060+)
- RAM: 16GB+
- Storage: 10GB+ SSD

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**:
   ```python
   # Giảm batch_size xuống 8 hoặc 4
   train_loader = DataLoader(train_dataset, batch_size=8)
   ```

2. **Model không load được**:
   ```python
   # Kiểm tra đường dẫn file best_model.pth
   # Đảm bảo model architecture trùng khớp
   ```

3. **CORS error trong web app**:
   ```python
   # Đã enable CORS trong Flask app
   # Kiểm tra URL API trong JavaScript
   ```


## 👥 Đóng góp

Dự án mở cho việc cải thiện và mở rộng:
- Thêm các mô hình segmentation khác (DeepLab, PSPNet)
- Cải thiện giao diện web
- Tối ưu hóa hiệu suất inference
- Thêm các metrics đánh giá khác

## 📄 License

Dự án được phát triển cho mục đích nghiên cứu và giáo dục.

---

**Liên hệ**: Để biết thêm thông tin chi tiết về dự án, vui lòng tham khảo các file code và documentation trong từng thư mục con.

