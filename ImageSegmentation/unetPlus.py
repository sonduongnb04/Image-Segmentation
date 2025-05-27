import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Dataset từ file .npy
class CamVidNPYDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.images = np.load(image_path)
        self.masks = np.load(mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = torch.tensor(augmented['mask'], dtype=torch.long)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomCrop(width=224, height=224),
    A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Dataloader
train_dataset = CamVidNPYDataset("X_train.npy", "Y_train.npy", transform=train_transform)
val_dataset = CamVidNPYDataset("X_val.npy", "Y_val.npy", transform=val_transform)
test_dataset = CamVidNPYDataset("X_test.npy", "Y_test.npy", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Khởi tạo mô hình UNet++
model = smp.UnetPlusPlus(
    encoder_name="resnet34",         # hoặc resnet18, efficientnet-b0...
    encoder_weights="imagenet",
    in_channels=3,
    classes=12                       # số lớp của CamVid
)

# Pixel Accuracy
def pixel_accuracy(preds, masks):
    correct = (preds == masks).float().sum()
    return correct / masks.numel()

# Mean IoU
def mean_iou(preds, masks, num_classes):
    ious = []
    preds, masks = preds.view(-1), masks.view(-1)
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        union = ((preds == cls) | (masks == cls)).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Dice Score
def dice_score(preds, masks, num_classes):
    dices = []
    preds, masks = preds.view(-1), masks.view(-1)
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        total = (preds == cls).sum().item() + (masks == cls).sum().item()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append(2 * intersection / total)
    return np.nanmean(dices)

# Validation
def evaluate(model, loader, combined_loss, device, num_classes):
    model.eval()
    loss, acc, miou, dice = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += combined_loss(out, y).item()
            preds = out.argmax(1)
            acc += pixel_accuracy(preds, y).item()
            miou += mean_iou(preds, y, num_classes)
            dice += dice_score(preds, y, num_classes)
    return loss / len(loader), acc / len(loader), miou / len(loader), dice / len(loader)

# Dice Loss cho segmentation multi-class
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model_unetPlus.pth', mode='max', metric_name='Dice + IoU'):
        """
        - patience: số epoch chờ nếu không cải thiện
        - delta: cải thiện nhỏ nhất để được tính
        - path: nơi lưu mô hình tốt nhất
        - mode: 'max' vì Dice + IoU càng cao càng tốt
        - metric_name: để hiển thị rõ metric đang theo dõi
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.mode = mode
        self.metric_name = metric_name

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_saved = None

    def __call__(self, current_score, model, epoch=None):
        score = -current_score if self.mode == 'min' else current_score

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model, epoch, current_score)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, epoch, current_score):
        if self.prev_saved and os.path.exists(self.prev_saved):
            try:
                os.remove(self.prev_saved)
                print(f"Đã xoá mô hình cũ: {self.prev_saved}")
            except:
                print(f"Không thể xoá file cũ: {self.prev_saved}")

        torch.save(model.state_dict(), self.path)
        print(f"Lưu mô hình tốt nhất tại epoch {epoch+1} với {self.metric_name} = {current_score:.4f}")
        self.prev_saved = self.path

# Huấn luyện mô hình
def train_model(model, train_loader, val_loader, num_epochs, device, num_classes):
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    def combined_loss(outputs, targets):
        return 0.7 * ce_loss(outputs, targets) + 0.3 * dice_loss(outputs, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    # Khởi tạo EarlyStopping theo Dice + IoU
    early_stopping = EarlyStopping(
        patience=10,
        path='best_model_unetPlus.pth',
        mode='max',
        metric_name='Dice + IoU'
    )
    
    train_losses, val_losses = [], []
    acc_list, miou_list, dice_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Trung bình loss của epoch
        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, acc, miou, dice = evaluate(model, val_loader, combined_loss, device, num_classes)

        # Ghi lại lịch sử
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc_list.append(acc)
        miou_list.append(miou)
        dice_list.append(dice)

        # Tính điểm kết hợp
        score_combined = 0.5 * dice + 0.5 * miou

        # In log
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Pixel Accuracy: {acc:.4f}")
        print(f"Mean IoU:       {miou:.4f}")
        print(f"Dice Score:     {dice:.4f}")
        print(f"Combined Score (0.5 * Dice + 0.5 * mIoU): {score_combined:.4f}")

        # Gọi EarlyStopping
        early_stopping(score_combined, model, epoch=epoch)

        if early_stopping.early_stop:
            print("Dừng sớm vì Dice + IoU không cải thiện.")
            break

    return train_losses, val_losses, acc_list, miou_list, dice_list


# Vẽ biểu đồ
def plot_metrics(train_losses, val_losses, acc, miou, dice):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()

    plt.subplot(222)
    plt.plot(epochs, acc, label='Pixel Accuracy', color='green')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy'); plt.legend()

    plt.subplot(223)
    plt.plot(epochs, miou, label='Mean IoU', color='purple')
    plt.xlabel('Epoch'); plt.ylabel('Mean IoU'); plt.title('Mean IoU'); plt.legend()

    plt.subplot(224)
    plt.plot(epochs, dice, label='Dice Score', color='red')
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title('Dice Score'); plt.legend()

    plt.tight_layout()
    plt.show()

# Huấn luyện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_losses, val_losses, acc_list, miou_list, dice_list = train_model(
    model, train_loader, val_loader, num_epochs=100, device=device, num_classes=12
)

plot_metrics(train_losses, val_losses, acc_list, miou_list, dice_list)

# Dự đoán & hiển thị kết quả
model.load_state_dict(torch.load("best_model_unetPlus.pth"))
model.to(device).eval()

num_samples = 5
indices = random.sample(range(len(test_dataset)), num_samples)

for i, idx in enumerate(indices):
    image, mask = test_dataset[idx]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

    
    img_show = image.permute(1, 2, 0).cpu().numpy()
    img_show = (img_show * 0.5 + 0.5) * 255
    img_show = np.clip(img_show, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_show)  
    plt.title("Test Image"); plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask.numpy(), cmap='jet')  
    plt.title("Ground Truth"); plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred.numpy(), cmap='jet')  
    plt.title("Prediction"); plt.axis('off')

    plt.tight_layout()
    plt.show()
