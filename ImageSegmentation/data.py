import os
import numpy as np
import cv2

# Hàm chuyển mask RGB sang mask nhãn
def rgb_to_label(mask_rgb, color_map):
    label_mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    for color, label in color_map.items():
        equality = np.all(mask_rgb == color, axis=-1)
        label_mask[equality] = label
    return label_mask

# Hàm load mask từ thư mục
def load_masks_as_label_array(mask_dir, color_map):
    mask_files = sorted(os.listdir(mask_dir))
    Y_list = []

    for fname in mask_files:
        fpath = os.path.join(mask_dir, fname)
        msk = cv2.imread(fpath, cv2.IMREAD_COLOR)  # đọc mask (BGR)
        if msk is None:
            print(f"Không đọc được mask: {fpath}")
            continue
        msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)  # chuyển mask sang RGB
        msk_label = rgb_to_label(msk_rgb, color_map)    # RGB sang label
        Y_list.append(msk_label)

    Y = np.array(Y_list, dtype=np.uint8)
    return Y  # (N, H, W)

# Load ảnh RGB bình thường
def load_images_as_array(image_dir):
    image_files = sorted(os.listdir(image_dir))
    X_list = []

    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            print(f"Không đọc được ảnh: {fpath}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_list.append(img_rgb)

    X = np.array(X_list, dtype=np.uint8)
    return X  # (N, H, W, 3)

# Hàm chính load dataset CamVid
def load_processed_Camvid_label(root_dir, color_map):
    train_img_dir = os.path.join(root_dir, "train", "images")
    train_msk_dir = os.path.join(root_dir, "train", "masks_color")

    val_img_dir = os.path.join(root_dir, "val", "images")
    val_msk_dir = os.path.join(root_dir, "val", "masks_color")

    test_img_dir = os.path.join(root_dir, "test", "images")
    test_msk_dir = os.path.join(root_dir, "test", "masks_color")

    # Load ảnh màu RGB
    X_train = load_images_as_array(train_img_dir)
    X_val = load_images_as_array(val_img_dir)
    X_test = load_images_as_array(test_img_dir)

    # Load masks dạng nhãn (N, H, W)
    Y_train = load_masks_as_label_array(train_msk_dir, color_map)
    Y_val = load_masks_as_label_array(val_msk_dir, color_map)
    Y_test = load_masks_as_label_array(test_msk_dir, color_map)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Colormap của bạn
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

# Chạy hàm load và lưu file npy
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed_Camvid_label("Camvid_processed", color_map)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print("Đã lưu X_train, Y_train, X_val, Y_val, X_test, Y_test dưới dạng .npy")