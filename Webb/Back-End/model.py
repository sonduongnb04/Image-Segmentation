import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm 
import matplotlib.pyplot as plt
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Tạo Dataset từ file .npy
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

# Tạo Dataloader 

train_dataset = CamVidNPYDataset("X_train.npy", "Y_train.npy", transform=train_transform)
val_dataset = CamVidNPYDataset("X_val.npy", "Y_val.npy", transform=val_transform)
test_dataset = CamVidNPYDataset("X_test.npy", "Y_test.npy", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Xây dựng mô hình 
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=12):  # 12 classes for CamVid
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.final(d1)


# Hàm tính Pixel Accuracy
def pixel_accuracy(preds, masks):
    correct = (preds == masks).float().sum()
    return correct / masks.numel()

# Hàm tính Mean IoU
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

# Hàm tính Dice Score
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

#Hàm đánh giá Validation

def evaluate(model,loader,combined_loss,device,num_classes):
    model.eval()
    loss,acc,miou,dice=0,0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x)
            loss+=combined_loss(out,y).item()
            preds=out.argmax(1)
            acc+=pixel_accuracy(preds,y).item()
            miou+=mean_iou(preds,y,num_classes)
            dice+=dice_score(preds,y,num_classes)
    return loss/len(loader),acc/len(loader),miou/len(loader),dice/len(loader)


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
    def __init__(self, patience=10, delta=0, path='best_model.pth', mode='min'):
      
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode  
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

        filename = self.path
        torch.save(model.state_dict(), filename)
        print(f"Lưu mô hình tốt nhất tại epoch {epoch+1} với val_loss = {current_score:.4f} -> {filename}")
        self.prev_saved = filename



# Huấn luyện cải tiến (tích hợp đánh giá và lưu mô hình tốt nhất)
def train_model(model, train_loader, val_loader, num_epochs, device, num_classes):
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    def combined_loss(outputs, targets):
        return 0.7 * ce_loss(outputs, targets) + 0.3 * dice_loss(outputs, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    early_stopping = EarlyStopping(patience=10, path='best_model.pth', mode='min')
    
    train_losses, val_losses = [], []
    acc_list, miou_list, dice_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, acc, miou, dice = evaluate(model, val_loader, combined_loss, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc_list.append(acc)
        miou_list.append(miou)
        dice_list.append(dice)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Acc: {acc:.4f}, mIoU: {miou:.4f}, Dice: {dice:.4f}")

        # lưu mô hình nếu val_loss tốt hơn
        early_stopping(val_loss, model, epoch=epoch)


        # (Tuỳ chọn) Dừng sớm nếu không cải thiện
        if early_stopping.early_stop:
            print("Dừng sớm vì val_loss không cải thiện.")
            break

    return train_losses, val_losses, acc_list, miou_list, dice_list


# Vẽ biểu đồ đánh giá
def plot_metrics(train_losses, val_losses, acc, miou, dice):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,8))

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

# Chạy mô hình huấn luyện hoàn chỉnh
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=12)

    train_losses, val_losses, acc_list, miou_list, dice_list = train_model(
        model, train_loader, val_loader, num_epochs=100, device=device, num_classes=12
    )

    plot_metrics(train_losses, val_losses, acc_list, miou_list, dice_list)

    # Tải và chạy thử mô hình tốt nhất
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device).eval()


    num_samples = 5
    indices = random.sample(range(len(test_dataset)), num_samples)

    for i, idx in enumerate(indices):
        image, mask = test_dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

        # Đoạn này đã sửa chuẩn (ảnh sẽ luôn hiển thị đúng)
        img_show = image.permute(1, 2, 0).cpu().numpy()
        img_show = (img_show * 0.5 + 0.5) * 255
        img_show = np.clip(img_show, 0, 255).astype(np.uint8)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_show)  # Đã sửa đúng
        plt.title("Test Image"); plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.numpy(), cmap='jet')  # Hiển thị mask
        plt.title("Ground Truth"); plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred.numpy(), cmap='jet')  # Hiển thị prediction
        plt.title("Prediction"); plt.axis('off')

        plt.tight_layout()
        plt.show()
