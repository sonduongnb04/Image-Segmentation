# Giao diện người dùng phân đoạn ảnh trực tuyến

## Giới thiệu
Đây là phần giao diện người dùng (front-end) cho ứng dụng phân đoạn ảnh trực tuyến, cho phép người dùng tải lên hình ảnh và nhận kết quả phân đoạn từ mô hình UNet.

## Cấu trúc thư mục
```
Front-End/
├── index.html      # Trang chính
├── styles.css      # Định dạng CSS
├── script.js       # Mã JavaScript cho tương tác
├── upload-icon.svg # Icon tải lên
└── README.md       # Tài liệu hướng dẫn
```

## Các tính năng
- Tải lên hình ảnh (kéo và thả hoặc chọn file)
- Xem trước hình ảnh trước khi xử lý
- Hỗ trợ nhiều ngôn ngữ (Tiếng Việt và Tiếng Anh)
- Hiển thị kết quả phân đoạn
- Khả năng tải xuống kết quả phân đoạn

## Cách sử dụng

### Thiết lập môi trường phát triển
1. Clone repository hoặc tải xuống mã nguồn
2. Mở thư mục Front-End trong trình biên tập mã (code editor)
3. Để phát triển cục bộ, bạn có thể sử dụng một server web tĩnh đơn giản:

Python:
```
python -m http.server
```

Node.js:
```
npx serve
```

### Liên kết với Back-End
Front-end hiện tại có chức năng giả lập kết quả phân đoạn. Để kết nối với mô hình UNet thực tế:

1. Thay thế hàm `mockSegmentation` trong `script.js` bằng một request API thực tế
2. Sửa đổi code để gửi hình ảnh đến API endpoint của back-end
3. Cập nhật cách xử lý phản hồi từ API

Ví dụ mã được cung cấp trong `script.js` có chú thích về nơi cần sửa đổi.

## Liên kết với back-end

Trong tình trạng hiện tại, front-end sử dụng một phân đoạn giả lập bằng JavaScript. Để kết nối với mô hình UNet thực:

1. Sửa đổi sự kiện click của nút xử lý trong `script.js`:
```javascript
// Thay đổi đoạn mã này
processBtn.addEventListener('click', function() {
    if (!uploadedImage) return;

    // Hiển thị loading
    previewContainer.hidden = true;
    loadingSection.hidden = false;

    // Tạo FormData để gửi file
    const formData = new FormData();
    formData.append('image', uploadedImage);

    // Gửi request đến API backend
    fetch('http://your-backend-url/api/segment', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hiển thị kết quả nhận được từ API
        segmentedImage.src = data.segmented_image_url;
        loadingSection.hidden = true;
        resultSection.hidden = false;
    })
    .catch(error => {
        console.error('Lỗi khi kết nối với backend:', error);
        alert('Có lỗi xảy ra khi xử lý ảnh. Vui lòng thử lại sau.');
        loadingSection.hidden = true;
        previewContainer.hidden = false;
    });
});
```

## Hỗ trợ đa ngôn ngữ
Website hỗ trợ Tiếng Việt và Tiếng Anh. Có thể thêm ngôn ngữ bằng cách cập nhật đối tượng `translations` trong `script.js`. 