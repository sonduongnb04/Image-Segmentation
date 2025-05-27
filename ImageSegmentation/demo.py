import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ==== UNet giống lúc train ====
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=12):
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
        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final(d1)

# ==== Tiền xử lý ảnh từ file ====
def load_image_from_file(image_path, size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return image_tensor, image.resize(size)

# ==== Dự đoán ====
def predict(model, image_tensor, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        pred_mask = output.argmax(1).squeeze().cpu().numpy()
    return pred_mask

# ==== Hiển thị kết quả ====
def show_prediction(image, pred_mask):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='jet')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==== MAIN ====
if __name__ == "__main__":
    print("Demo dự đoán phân đoạn ảnh đơn với UNet")

    IMAGE_PATH = "0001TP_008550.png"  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=12)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device).eval()

    image_tensor, original_image = load_image_from_file(IMAGE_PATH)
    pred_mask = predict(model, image_tensor, device)
    show_prediction(original_image, pred_mask)
