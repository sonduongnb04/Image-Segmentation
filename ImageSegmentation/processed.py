import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

##############################################################################
# BẢNG MÀU CAMVID (RGB) -> CLASS_ID
##############################################################################
color_map = {
    (128, 128, 128): 0,   # Sky
    (128, 0, 0): 1,       # Building
    (192, 192, 128): 2,   # Pole
    (128, 64, 128): 3,    # Road
    (60, 40, 222): 4,     # Pavement
    (128, 128, 0): 5,     # Tree
    (192, 128, 128): 6,   # SignSymbol
    (64, 64, 128): 7,     # Fence
    (64, 0, 128): 8,      # Car
    (64, 64, 0): 9,       # Pedestrian
    (0, 128, 192): 10,    # Bicyclist
    (0, 0, 0): 11         # Unlabeled
}

##############################################################################
# BẢNG MÀU HIỂN THỊ (CLASS_ID) -> (R, G, B)
##############################################################################
id_to_color = {
    0:  (128, 128, 128),  # Sky
    1:  (128, 0, 0),      # Building
    2:  (192, 192, 128),  # Pole
    3:  (128, 64, 128),   # Road
    4:  (60, 40, 222),    # Pavement
    5:  (128, 128, 0),    # Tree
    6:  (192, 128, 128),  # SignSymbol
    7:  (64, 64, 128),    # Fence
    8:  (64, 0, 128),     # Car
    9:  (64, 64, 0),      # Pedestrian
    10: (0, 128, 192),    # Bicyclist
    11: (0, 0, 0)         # Unlabeled
}

def color_to_integer_mask(mask_color, color_map):
    """
    mask_color: (H, W, 3) BGR
    color_map: dict {(R,G,B): class_id}
    Trả về: mask_int (H, W) uint8, mỗi pixel = class_id (0..11)
    """
    h, w = mask_color.shape[:2]
    mask_int = np.zeros((h, w), dtype=np.uint8)
    
    for rgb_color, class_id in color_map.items():
        bgr_color = tuple(reversed(rgb_color))  # (R,G,B) -> (B,G,R)
        match = np.all(mask_color == bgr_color, axis=-1)
        mask_int[match] = class_id
    return mask_int

def integer_mask_to_color(mask_int, id_to_color):
    """
    mask_int: (H, W), pixel = class_id
    id_to_color: dict {class_id: (R,G,B)}
    Trả về: color_mask_bgr (H, W, 3) BGR
    """
    h, w = mask_int.shape
    color_mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb_color in id_to_color.items():
        bgr_color = tuple(reversed(rgb_color))
        color_mask_bgr[mask_int == class_id] = bgr_color
    return color_mask_bgr

##############################################################################
# PIPELINE ALBUMENTATIONS (có Normalize)
##############################################################################
transform = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),  
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
], additional_targets={'mask': 'mask'})

def preprocess_image(image_bgr, mask_color_bgr, transform):
    """
    - image_bgr: ảnh gốc (BGR)
    - mask_color_bgr: mask màu (BGR)
    - transform: Albumentations
    
    Trả về:
      image_aug: float32, (H, W, 3), đã Normalize
      mask_aug:  int64, (H, W), integer mask
    """
    # 1) BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2) mask màu -> mask integer
    mask_int = color_to_integer_mask(mask_color_bgr, color_map)
    
    # 3) Albumentations
    augmented = transform(image=image_rgb, mask=mask_int)
    image_aug = augmented['image']  # float32, normalized
    mask_aug = augmented['mask']    # int64
    
    return image_aug, mask_aug

##############################################################################
# HÀM XỬ LÝ 1 CẶP THƯ MỤC (ảnh, mask) -> LƯU VÀO 2 THƯ MỤC (images, masks_color)
##############################################################################
def process_dataset(image_dir, mask_dir, out_image_dir, out_mask_dir,
                    transform, id_to_color, mean, std):
    """
    - image_dir: thư mục ảnh gốc (VD: 'Dataset/train')
    - mask_dir: thư mục mask màu gốc (VD: 'Dataset/train_labels')
    - out_image_dir: nơi lưu ảnh đã augment + de-normalize
    - out_mask_dir: nơi lưu mask màu hiển thị (thay vì mask integer)
    - transform: pipeline Albumentations
    - id_to_color: dict {class_id: (R,G,B)}
    - mean, std: dùng để khử chuẩn hóa
    
    Hàm này sẽ:
      1) Duyệt tất cả file ảnh trong image_dir
      2) Ghép tên mask tương ứng (VD: baseName + '_L' + ext)
      3) Đọc ảnh + mask -> preprocess_image -> de-normalize ảnh -> lưu
      4) Chuyển mask integer -> mask màu -> lưu
    """
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_dir))
    
    for img_name in image_files:
        base_name, ext = os.path.splitext(img_name)
        # Giả sử mask_name = base_name + '_L' + ext
        mask_name = base_name + "_L" + ext
        
        img_path = os.path.join(image_dir, img_name)
        msk_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(msk_path):
            print(f"Không tìm thấy mask cho {img_name}, bỏ qua.")
            continue
        
        image_bgr = cv2.imread(img_path)
        mask_bgr  = cv2.imread(msk_path)
        
        if image_bgr is None or mask_bgr is None:
            print(f"Không thể đọc {img_path} hoặc {msk_path}, bỏ qua.")
            continue
        
        # Gọi hàm tiền xử lý
        image_aug, mask_aug = preprocess_image(image_bgr, mask_bgr, transform)
        
        # 1) De-normalize ảnh
        image_denorm = (image_aug * std + mean).clip(0, 1)  # [0..1]
        image_255 = (image_denorm * 255).astype(np.uint8)   # [0..255]
        # RGB -> BGR để lưu
        image_bgr_out = cv2.cvtColor(image_255, cv2.COLOR_RGB2BGR)
        
        out_image_path = os.path.join(out_image_dir, img_name)
        cv2.imwrite(out_image_path, image_bgr_out)
        
        # 2) Mask integer -> mask màu
        color_mask_bgr = integer_mask_to_color(mask_aug, id_to_color)
        
        out_mask_path = os.path.join(out_mask_dir, mask_name)
        cv2.imwrite(out_mask_path, color_mask_bgr)
        
        print(f"Đã xử lý & lưu: {img_name} -> {mask_name}")

##############################################################################
# CHẠY CHO 3 CẶP (train, val, test)
##############################################################################
if __name__ == "__main__":
    # Đường dẫn gốc
    data_root = "Camvid"
    
    # Các thư mục ảnh + mask
    train_image_dir = os.path.join(data_root, "train")
    train_mask_dir  = os.path.join(data_root, "train_labels")
    
    val_image_dir = os.path.join(data_root, "val")
    val_mask_dir  = os.path.join(data_root, "val_labels")
    
    test_image_dir = os.path.join(data_root, "test")
    test_mask_dir  = os.path.join(data_root, "test_labels")
    
    # Thư mục xuất
    out_root = "Camvid_processed"
    
    # Tạo đường dẫn đầu ra
    train_out_image_dir = os.path.join(out_root, "train", "images")
    train_out_mask_dir  = os.path.join(out_root, "train", "masks_color")
    
    val_out_image_dir = os.path.join(out_root, "val", "images")
    val_out_mask_dir  = os.path.join(out_root, "val", "masks_color")
    
    test_out_image_dir = os.path.join(out_root, "test", "images")
    test_out_mask_dir  = os.path.join(out_root, "test", "masks_color")
    
    # mean/std để khử chuẩn hóa
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 1) Xử lý train
    print("=== Xử lý TRAIN ===")
    process_dataset(train_image_dir, train_mask_dir,
                    train_out_image_dir, train_out_mask_dir,
                    transform, id_to_color, mean, std)
    
    # 2) Xử lý val
    print("\n=== Xử lý VAL ===")
    process_dataset(val_image_dir, val_mask_dir,
                    val_out_image_dir, val_out_mask_dir,
                    transform, id_to_color, mean, std)
    
    # 3) Xử lý test
    print("\n=== Xử lý TEST ===")
    process_dataset(test_image_dir, test_mask_dir,
                    test_out_image_dir, test_out_mask_dir,
                    transform, id_to_color, mean, std)
    
    print("\nHoàn tất!")
