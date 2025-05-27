from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import base64
import os
from flask_cors import CORS
from model import UNet
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Bật CORS cho tất cả các route

# Định nghĩa color map cho các lớp phân đoạn
COLOR_MAP = {
    0: (128, 128, 128),  # Bầu trời (Sky)
    1: (128, 0, 0),      # Tòa nhà (Building)
    2: (192, 192, 128),  # Cột (Pole)
    3: (128, 64, 128),   # Đường (Road)
    4: (60, 40, 222),    # Vỉa hè (Pavement)
    5: (128, 128, 0),    # Cây (Tree)
    6: (192, 128, 128),  # Biển báo (SignSymbol)
    7: (64, 64, 128),    # Hàng rào (Fence)
    8: (64, 0, 128),     # Xe hơi (Car)
    9: (64, 64, 0),      # Người đi bộ (Pedestrian)
    10: (0, 128, 192),   # Người đi xe đạp (Bicyclist)
    11: (0, 0, 0)        # Không xác định (Unlabeled)
}

# Khởi tạo model và tải weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    global model
    model = UNet(in_channels=3, out_channels=12)
    try:
        model_path = 'best_model.pth'
        if not os.path.exists(model_path):
            model_path = '../best_model.pth'  # Thử tìm ở thư mục cha
        print(f"Đang tải mô hình từ {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Tải mô hình thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {str(e)}")
        print("Vui lòng đảm bảo file best_model.pth đã được đặt trong thư mục Back-End hoặc thư mục gốc.")
        raise

def preprocess_image(image):
    # Thay đổi kích thước ảnh về đầu vào mong muốn
    image = cv2.resize(image, (256, 256))
    # Chuyển đổi sang dạng float và chuẩn hóa
    image = image.astype(np.float32) / 255.0
    # Chuẩn hóa với các giá trị giống khi huấn luyện
    image = (image - 0.5) / 0.5
    # Chuyển thành tensor PyTorch và thêm chiều batch
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return image.to(device)

def create_colored_mask(prediction):
    # Tạo mask phân đoạn có màu
    colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        colored_mask[prediction == class_idx] = color
    return colored_mask

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Không có ảnh được cung cấp"}), 400
    
    try:
        # Đọc ảnh từ request
        file = request.files['image']
        img_data = file.read()
        
        # Chuyển sang định dạng OpenCV
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
        
        # Lưu kích thước gốc để resize lại sau
        original_h, original_w = img.shape[:2]
        
        # Tiền xử lý ảnh
        input_tensor = preprocess_image(img)
        
        # Thực hiện dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(1).squeeze().cpu().numpy()
        
        # Tạo mask phân đoạn có màu
        colored_mask = create_colored_mask(prediction)
        
        # Resize trở lại kích thước gốc
        colored_mask = cv2.resize(colored_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Chuyển mask thành chuỗi base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "segmented_image": f"data:image/png;base64,{img_str}"
        })
    
    except Exception as e:
        print(f"Lỗi trong quá trình phân đoạn: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 