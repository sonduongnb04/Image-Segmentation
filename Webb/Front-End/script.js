// Ghi chú: File này xử lý logic cho trang tải ảnh lên (index.html)
document.addEventListener('DOMContentLoaded', function () {
    // Lấy các phần tử trên trang web mà chúng ta cần tương tác
    const uploadArea = document.getElementById('upload-area'); // Khu vực kéo thả file
    const fileInput = document.getElementById('file-input'); // Input ẩn để chọn file
    const selectFileBtn = document.getElementById('select-file-btn'); // Nút "Chọn tệp"
    const processBtn = document.getElementById('process-btn'); // Nút "Xử lý"
    const changeImageBtn = document.getElementById('change-image-btn'); // Nút "Đổi ảnh"
    const imagePreviewContainer = document.getElementById('image-preview-container'); // Vùng chứa ảnh xem trước
    const previewImage = document.getElementById('preview-image'); // Thẻ <img> để xem trước
    const uploadSection = document.getElementById('upload-section'); // Toàn bộ phần tải lên
    const vnLangBtn = document.getElementById('vn-lang'); // Nút chọn tiếng Việt
    const enLangBtn = document.getElementById('en-lang'); // Nút chọn tiếng Anh

    // Biến để lưu file ảnh người dùng chọn
    let uploadedImage = null;
    // Biến để lưu ảnh dưới dạng Data URL (dùng để hiển thị và gửi đi nếu cần)
    let uploadedImageDataUrl = null;

    // Địa chỉ của backend API (thay đổi nếu backend chạy ở địa chỉ khác)
    const API_URL = 'http://localhost:5000';

    // Lưu trữ các chuỗi dịch cho đa ngôn ngữ
    let currentLang = localStorage.getItem('language') || 'vi'; // Lấy ngôn ngữ đã lưu hoặc mặc định là 'vi'
    const translations = {
        vi: { /* Bản dịch tiếng Việt */
            title: 'Phân đoạn ảnh giao thông trực tuyến',
            upload: 'Tải ảnh lên',
            selectFile: 'Chọn tệp',
            process: 'Xử lý',
            processing: 'Đang xử lý...',
            changeImage: 'Đổi ảnh',
            copyright: '© 2025 Phân đoạn ảnh giao thông trực tuyến. Dự án nghiên cứu.',
            error: 'Có lỗi xảy ra khi xử lý ảnh. Vui lòng thử lại sau.',
            invalidFile: 'Vui lòng chọn file ảnh hợp lệ!',
            serverError: 'Không thể kết nối đến server. Đang sử dụng chế độ giả lập.',
            backendNotReady: 'Backend chưa sẵn sàng. Đang sử dụng chế độ giả lập.',
            previewAlt: 'Xem trước Ảnh'
        },
        en: { /* Bản dịch tiếng Anh */
            title: 'Online Traffic Image Segmentation',
            upload: 'Upload image',
            selectFile: 'Select file',
            process: 'Process',
            processing: 'Processing...',
            changeImage: 'Change image',
            copyright: '© 2025 Online Traffic Image Segmentation. Research project.',
            error: 'An error occurred while processing the image. Please try again later.',
            invalidFile: 'Please select a valid image file!',
            serverError: 'Cannot connect to server. Using fallback mode.',
            backendNotReady: 'Backend is not ready. Using fallback mode.',
            previewAlt: 'Image Preview'
        }
    };

    // Biến kiểm tra trạng thái kết nối với backend
    let isBackendAvailable = false;

    // --- Các hàm xử lý chính ---

    // Hàm cập nhật ngôn ngữ hiển thị trên trang
    function setLanguage(lang) {
        currentLang = lang;
        localStorage.setItem('language', lang); // Lưu ngôn ngữ được chọn vào localStorage

        // Đặt/xóa class 'active' cho nút ngôn ngữ tương ứng
        if (vnLangBtn) vnLangBtn.classList.toggle('active', lang === 'vi');
        if (enLangBtn) enLangBtn.classList.toggle('active', lang === 'en');

        // Cập nhật nội dung text của các phần tử theo ngôn ngữ đã chọn
        document.title = translations[lang].title;
        const headerH1 = document.querySelector('header h1');
        if (headerH1) headerH1.textContent = translations[lang].title;
        const uploadH2 = document.querySelector('.upload-section h2');
        if (uploadH2) uploadH2.textContent = translations[lang].upload;
        if (selectFileBtn) selectFileBtn.textContent = translations[lang].selectFile;
        if (processBtn && !processBtn.disabled) processBtn.textContent = translations[lang].process;
        if (changeImageBtn) changeImageBtn.textContent = translations[lang].changeImage;
        const footerP = document.querySelector('footer p');
        if (footerP) footerP.textContent = translations[lang].copyright;
        if (previewImage) previewImage.alt = translations[lang].previewAlt; // Cập nhật alt text cho ảnh
    }

    // Hàm kiểm tra kết nối đến backend API
    function checkBackendConnection() {
        return fetch(`${API_URL}/api/health`) // Gửi request đến endpoint /api/health
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok'); // Báo lỗi nếu response không thành công
                return response.json(); // Chuyển response thành JSON
            })
            .then(data => {
                console.log('Backend status:', data); // In trạng thái backend ra console
                isBackendAvailable = data.status === 'ok'; // Cập nhật biến trạng thái
                return isBackendAvailable;
            })
            .catch(error => {
                console.error('Backend connection error:', error); // In lỗi nếu không kết nối được
                isBackendAvailable = false;
                return false; // Trả về false khi có lỗi
            });
    }

    // Hàm ngăn chặn hành vi mặc định của trình duyệt (cần cho kéo thả)
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Hàm thêm hiệu ứng highlight khi kéo file vào khu vực upload
    function highlight() {
        if (uploadArea) uploadArea.classList.add('highlight');
    }

    // Hàm xóa hiệu ứng highlight khi kéo file ra hoặc thả file
    function unhighlight() {
        if (uploadArea) uploadArea.classList.remove('highlight');
    }

    // Hàm xử lý khi thả file vào khu vực upload
    function handleDrop(e) {
        preventDefaults(e); // Ngăn hành vi mặc định
        unhighlight(); // Xóa highlight
        const dt = e.dataTransfer; // Lấy dữ liệu được kéo thả
        const files = dt.files; // Lấy danh sách file
        if (files.length) {
            handleFiles(files); // Xử lý file đầu tiên
        }
    }

    // Hàm xử lý file được chọn (qua kéo thả hoặc nút chọn)
    function handleFiles(files) {
        uploadedImage = files[0]; // Lấy file đầu tiên
        // Kiểm tra xem có phải file ảnh hợp lệ không
        if (!uploadedImage || !uploadedImage.type.match('image.*')) {
            alert(translations[currentLang].invalidFile); // Thông báo lỗi file không hợp lệ
            resetUpload(); // Reset lại giao diện
            return;
        }

        // Đọc nội dung file ảnh để hiển thị preview
        const reader = new FileReader();
        reader.readAsDataURL(uploadedImage);
        reader.onload = function () {
            uploadedImageDataUrl = reader.result; // Lưu ảnh dưới dạng Data URL
            if (previewImage) previewImage.src = uploadedImageDataUrl; // Hiển thị ảnh xem trước
            if (imagePreviewContainer) imagePreviewContainer.hidden = false; // Hiện khung xem trước
            if (selectFileBtn) selectFileBtn.hidden = true; // Ẩn nút "Chọn tệp"
            if (uploadArea) uploadArea.classList.add('has-image'); // Thêm class để style (nếu cần)
            if (processBtn) processBtn.disabled = false; // Bật nút "Xử lý"
            if (changeImageBtn) changeImageBtn.hidden = false; // Hiện nút "Đổi ảnh"
        };
        reader.onerror = function () { // Xử lý nếu có lỗi khi đọc file
            console.error("Error reading file.");
            alert(translations[currentLang].error);
            resetUpload();
        }
    }

    // Hàm bật/tắt và đặt lại text cho các nút điều khiển
    function enableButtons() {
        if (processBtn) {
            processBtn.disabled = true; // Nút xử lý bị vô hiệu hóa ban đầu
            processBtn.textContent = translations[currentLang].process;
        }
        if (changeImageBtn) {
            changeImageBtn.hidden = true; // Nút đổi ảnh bị ẩn ban đầu
            changeImageBtn.disabled = false; // Đảm bảo nút đổi ảnh không bị vô hiệu hóa
        }
        // Không cần reset selectFileBtn ở đây vì resetUpload sẽ xử lý
    }

    // Hàm reset giao diện về trạng thái ban đầu
    function resetUpload() {
        uploadedImage = null; // Xóa file đã lưu
        uploadedImageDataUrl = null; // Xóa data URL
        if (fileInput) fileInput.value = ''; // Reset input file để có thể chọn lại cùng file
        if (previewImage) previewImage.src = ''; // Xóa ảnh xem trước
        if (imagePreviewContainer) imagePreviewContainer.hidden = true; // Ẩn khung xem trước
        if (selectFileBtn) selectFileBtn.hidden = false; // Hiện lại nút "Chọn tệp"
        if (uploadArea) {
            uploadArea.classList.remove('has-image'); // Xóa class nếu có
        }
        enableButtons(); // Đặt lại trạng thái các nút xử lý/đổi ảnh
    }

    // Hàm gửi ảnh đến backend để xử lý phân đoạn
    function processImageWithBackend() {
        const formData = new FormData(); // Tạo đối tượng FormData để gửi file
        formData.append('image', uploadedImage);

        fetch(`${API_URL}/api/segment`, { // Gọi API segment
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) throw new Error(`Network response was not ok (${response.status})`);
                return response.json();
            })
            .then(data => {
                if (data.success && data.segmented_image) { // Nếu backend xử lý thành công và trả về ảnh
                    localStorage.setItem('segmentedImage', data.segmented_image); // Lưu ảnh phân đoạn vào localStorage
                    if (uploadedImageDataUrl) localStorage.setItem('originalImage', uploadedImageDataUrl); // Lưu ảnh gốc (data URL) vào localStorage
                    window.location.href = 'edit.html'; // Chuyển hướng sang trang chỉnh sửa
                } else {
                    throw new Error(data.error || 'Invalid data received from backend'); // Báo lỗi nếu dữ liệu không hợp lệ
                }
            })
            .catch(error => { // Xử lý lỗi khi gọi API
                console.error('Error processing with backend:', error);
                alert(translations[currentLang].error);
                resetUpload(); // Reset giao diện khi có lỗi
            });
    }

    // Hàm giả lập xử lý phân đoạn (khi không kết nối được backend)
    function mockSegmentation(imageUrl) {
        // Kiểm tra imageUrl có hợp lệ không
        if (!imageUrl || typeof imageUrl !== 'string') {
            console.error("Invalid image URL for mock segmentation.");
            alert(translations[currentLang].error);
            resetUpload();
            return;
        }
        try {
            const img = new Image();
            img.crossOrigin = "Anonymous"; // Cần thiết nếu ảnh từ nguồn khác domain (tránh lỗi tainted canvas)
            img.src = imageUrl;
            img.onload = function () {
                // Kiểm tra kích thước ảnh sau khi tải
                if (img.naturalWidth === 0 || img.naturalHeight === 0) {
                    console.error("Image loaded with zero dimensions.");
                    alert(translations[currentLang].error);
                    resetUpload();
                    return;
                }
                // Vẽ ảnh lên canvas ẩn
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                ctx.drawImage(img, 0, 0);
                try {
                    // Lấy dữ liệu pixel từ canvas
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data;
                    // Áp dụng bộ lọc màu giả lập phân đoạn (ví dụ đơn giản)
                    for (let i = 0; i < data.length; i += 4) { const avg = (data[i] + data[i + 1] + data[i + 2]) / 3; if (avg > 200) { data[i] = 255; data[i + 1] = 100; data[i + 2] = 100; } else if (avg > 150) { data[i] = 100; data[i + 1] = 255; data[i + 2] = 100; } else if (avg > 100) { data[i] = 100; data[i + 1] = 100; data[i + 2] = 255; } else if (avg > 50) { data[i] = 255; data[i + 1] = 255; data[i + 2] = 0; } else { data[i] = 128; data[i + 1] = 0; data[i + 2] = 128; } }
                    ctx.putImageData(imageData, 0, 0); // Vẽ lại dữ liệu đã thay đổi lên canvas
                    const segmentedDataUrl = canvas.toDataURL('image/png'); // Lấy ảnh từ canvas dưới dạng Data URL

                    // Lưu kết quả và chuyển trang
                    localStorage.setItem('segmentedImage', segmentedDataUrl);
                    localStorage.setItem('originalImage', imageUrl);
                    window.location.href = 'edit.html';
                } catch (e) {
                    // Xử lý lỗi liên quan đến canvas (ví dụ: tainted canvas)
                    console.error("Error processing canvas data (maybe tainted canvas?):", e);
                    alert(translations[currentLang].error + " (Canvas Error)");
                    resetUpload();
                }
            };
            img.onerror = function (e) { // Xử lý lỗi khi không tải được ảnh
                console.error("Error loading image for mock segmentation:", e);
                alert(translations[currentLang].error);
                resetUpload();
            };
        } catch (error) { // Bắt các lỗi khác trong quá trình giả lập
            console.error("Error in mock segmentation:", error);
            alert(translations[currentLang].error);
            resetUpload();
        }
    }

    // --- Gán các sự kiện cho các phần tử --- 

    // Sự kiện click cho nút chọn ngôn ngữ
    if (vnLangBtn) vnLangBtn.addEventListener('click', () => setLanguage('vi'));
    if (enLangBtn) enLangBtn.addEventListener('click', () => setLanguage('en'));

    // Gán sự kiện kéo thả cho khu vực upload
    ['dragenter', 'dragover'].forEach(eventName => {
        if (uploadArea) uploadArea.addEventListener(eventName, (e) => {
            preventDefaults(e);
            highlight();
        }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        if (uploadArea) uploadArea.addEventListener(eventName, (e) => {
            preventDefaults(e);
            unhighlight();
        }, false);
    });
    if (uploadArea) uploadArea.addEventListener('drop', handleDrop, false);

    // Sự kiện click cho các nút
    if (selectFileBtn) selectFileBtn.addEventListener('click', () => { if (fileInput) fileInput.click(); });
    if (fileInput) fileInput.addEventListener('change', function () { if (this.files.length) handleFiles(this.files); });
    if (changeImageBtn) changeImageBtn.addEventListener('click', resetUpload);
    if (processBtn) processBtn.addEventListener('click', function () {
        // Chỉ xử lý nếu đã có ảnh được chọn
        if (!uploadedImage || !uploadedImageDataUrl) return;

        // Vô hiệu hóa các nút và cập nhật text nút xử lý
        processBtn.disabled = true;
        processBtn.textContent = translations[currentLang].processing || 'Processing...';
        if (changeImageBtn) changeImageBtn.disabled = true;

        // Kiểm tra backend hoặc fallback sang mock sau 3 giây
        const backendCheckPromise = checkBackendConnection();
        const timeoutPromise = new Promise((resolve) => { setTimeout(() => resolve(false), 3000); });

        Promise.race([backendCheckPromise, timeoutPromise]) // Chạy song song việc kiểm tra và timeout
            .then(available => {
                if (available) { // Nếu backend sẵn sàng
                    processImageWithBackend();
                } else { // Nếu không kết nối được hoặc quá 3 giây
                    console.warn(translations[currentLang].backendNotReady || translations[currentLang].serverError);
                    mockSegmentation(uploadedImageDataUrl); // Dùng hàm giả lập
                }
            })
            .catch(error => { // Xử lý lỗi không mong muốn trong quá trình kiểm tra
                console.error("Error during backend check/timeout race:", error);
                mockSegmentation(uploadedImageDataUrl); // Vẫn fallback sang mock
            });
    });


    // --- Khởi tạo trạng thái ban đầu khi tải trang --- 
    setLanguage(currentLang); // Đặt ngôn ngữ ban đầu
    checkBackendConnection(); // Kiểm tra trạng thái backend
    resetUpload(); // Đặt giao diện về trạng thái ban đầu
});