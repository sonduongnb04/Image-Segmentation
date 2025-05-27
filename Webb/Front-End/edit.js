// Ghi chú: File này xử lý logic cho trang chỉnh sửa ảnh (edit.html)
document.addEventListener('DOMContentLoaded', function () {
    // Lấy các phần tử DOM cần thiết trên trang chỉnh sửa
    const originalImage = document.getElementById('original-image'); // Thẻ <img> hiển thị ảnh gốc
    const editImage = document.getElementById('edit-image'); // Thẻ <img> hiển thị ảnh phân đoạn
    const originalTitle = document.getElementById('original-title'); // Tiêu đề của ảnh gốc
    const segmentedTitle = document.getElementById('segmented-title'); // Tiêu đề của ảnh phân đoạn
    const downloadEditBtn = document.getElementById('download-edit-btn'); // Nút "Tải về"
    const zoomBtn = document.getElementById('zoom-btn'); // Nút "Thu phóng"
    const clearBtn = document.getElementById('clear-btn'); // Nút "Xóa bỏ"
    const backBtn = document.getElementById('back-btn'); // Nút/Link "Trở lại" (trong file này là thẻ <a>)
    const backText = document.getElementById('back-text'); // Phần text của nút "Trở lại"
    const zoomText = document.getElementById('zoom-text'); // Phần text của nút "Thu phóng"
    const clearText = document.getElementById('clear-text'); // Phần text của nút "Xóa bỏ"
    const vnLangBtn = document.getElementById('vn-lang'); // Nút chọn tiếng Việt
    const enLangBtn = document.getElementById('en-lang'); // Nút chọn tiếng Anh

    // Phần tử modal
    const zoomModal = document.getElementById('zoom-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // Đảm bảo modal được ẩn khi trang tải
    if (zoomModal) {
        zoomModal.style.display = 'none';
    }

    // Lấy ảnh từ localStorage (đã được lưu từ trang index.html)
    const originalImageUrl = localStorage.getItem('originalImage');
    const segmentedImageUrl = localStorage.getItem('segmentedImage');

    // Kiểm tra và hiển thị ảnh nếu có
    if (originalImageUrl && originalImage) {
        originalImage.src = originalImageUrl;
    }

    if (segmentedImageUrl && editImage) {
        editImage.src = segmentedImageUrl;
    }

    // Nếu không tìm thấy cả hai ảnh, chuyển hướng về trang chính
    if ((!originalImageUrl || !segmentedImageUrl) && (originalImage || editImage)) {
        console.warn("Không tìm thấy đủ ảnh trong localStorage. Chuyển về trang chủ.");
        window.location.href = 'index.html';
        return; // Dừng thực thi script vì không có ảnh để hiển thị
    }

    // Cập nhật copyright year cho nhất quán
    const currentYear = new Date().getFullYear();

    // Định nghĩa các chuỗi dịch
    let currentLang = localStorage.getItem('language') || 'vi';
    const translations = {
        vi: { /* Bản dịch tiếng Việt */
            title: 'Chỉnh sửa ảnh phân đoạn',
            header: 'Phân đoạn ảnh giao thông trực tuyến', // Giữ header nhất quán
            back: 'Trở lại',
            zoom: 'Thu phóng',
            zoomIn: 'Phóng to', // Có thể dùng nếu muốn thay đổi text khi zoom
            zoomOut: 'Thu nhỏ',
            clear: 'Xóa bỏ',
            download: 'Tải về',
            originalTitle: 'Ảnh gốc',
            segmentedTitle: 'Ảnh phân đoạn',
            modalAlt: 'Ảnh phân đoạn phóng to',
            copyright: `© ${currentYear} Phân đoạn ảnh giao thông trực tuyến. Dự án nghiên cứu.`
        },
        en: { /* Bản dịch tiếng Anh */
            title: 'Edit Segmented Image',
            header: 'Online Traffic Image Segmentation', // Keep header consistent
            back: 'Back',
            zoom: 'Zoom',
            zoomIn: 'Zoom In',
            zoomOut: 'Zoom Out',
            clear: 'Clear',
            download: 'Download',
            originalTitle: 'Original Image',
            segmentedTitle: 'Segmented Image',
            modalAlt: 'Enlarged Segmented Image',
            copyright: `© ${currentYear} Online Traffic Image Segmentation. Research project.`
        }
    };

    // Hàm cập nhật ngôn ngữ hiển thị
    function setLanguage(lang) {
        currentLang = lang;
        localStorage.setItem('language', lang);

        if (vnLangBtn) vnLangBtn.classList.toggle('active', lang === 'vi');
        if (enLangBtn) enLangBtn.classList.toggle('active', lang === 'en');

        // Cập nhật các text trên trang
        document.title = translations[lang].title;
        const headerH1 = document.querySelector('header h1');
        if (headerH1) headerH1.textContent = translations[lang].header;
        if (backText) backText.textContent = translations[lang].back;
        if (zoomText) zoomText.textContent = translations[lang].zoom;
        if (clearText) clearText.textContent = translations[lang].clear;
        if (downloadEditBtn) downloadEditBtn.textContent = translations[lang].download;
        if (originalTitle) originalTitle.textContent = translations[lang].originalTitle;
        if (segmentedTitle) segmentedTitle.textContent = translations[lang].segmentedTitle;
        if (modalImage) modalImage.alt = translations[lang].modalAlt;
        const footerP = document.querySelector('footer p');
        if (footerP) footerP.textContent = translations[lang].copyright;
    }

    // Hàm mở Modal
    function openZoomModal() {
        if (zoomModal && modalImage && segmentedImageUrl) {
            modalImage.src = segmentedImageUrl;
            zoomModal.style.display = 'flex'; // Sử dụng flex thay vì block để căn chỉnh
            // Thêm timeout nhỏ để trigger animation
            setTimeout(() => {
                zoomModal.classList.add('show');
            }, 10);
        }
    }

    // Hàm đóng Modal
    function closeZoomModal() {
        if (zoomModal) {
            zoomModal.classList.remove('show');
            // Đợi animation kết thúc rồi mới ẩn modal và xóa src
            setTimeout(() => {
                zoomModal.style.display = 'none';
                if (modalImage) modalImage.src = '';
            }, 300); // Đợi transition (0.3s) hoàn thành
        }
    }

    // --- Gán sự kiện --- 

    // Sự kiện click cho nút chọn ngôn ngữ
    if (vnLangBtn) vnLangBtn.addEventListener('click', () => setLanguage('vi'));
    if (enLangBtn) enLangBtn.addEventListener('click', () => setLanguage('en'));

    // Sự kiện click cho nút "Thu phóng" - mở modal thay vì zoom ảnh nhỏ
    if (zoomBtn) zoomBtn.addEventListener('click', openZoomModal);

    // Sự kiện đóng modal
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeZoomModal);
    if (zoomModal) zoomModal.addEventListener('click', function (event) {
        // Đóng modal nếu click vào nền đen bên ngoài modal-content
        if (event.target === zoomModal) {
            closeZoomModal();
        }
    });

    // Thêm sự kiện phím ESC để đóng modal
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && zoomModal && zoomModal.style.display !== 'none') {
            closeZoomModal();
        }
    });

    // Sự kiện click cho nút "Xóa bỏ"
    if (clearBtn) clearBtn.addEventListener('click', function () {
        // Xóa dữ liệu ảnh đã lưu trữ
        localStorage.removeItem('segmentedImage');
        localStorage.removeItem('originalImage');
        // Quay về trang tải ảnh ban đầu
        window.location.href = 'index.html';
    });

    // Sự kiện click cho nút "Tải về"
    if (downloadEditBtn) downloadEditBtn.addEventListener('click', function () {
        if (!editImage || !editImage.src) return; // Kiểm tra xem có ảnh để tải không

        // Tạo một thẻ <a> ẩn để kích hoạt việc tải xuống
        const link = document.createElement('a');
        link.download = 'segmented-image.png'; // Tên file tải về mặc định
        link.href = editImage.src; // Đường dẫn là data URL của ảnh
        link.click(); // Kích hoạt việc tải xuống

        // Chuyển hướng sang trang thông báo tải xong sau một khoảng trễ nhỏ
        // để trình duyệt kịp xử lý việc tải file
        setTimeout(() => {
            window.location.href = 'download-complete.html';
        }, 150); // Tăng nhẹ thời gian chờ
    });

    // --- Khởi tạo --- 
    setLanguage(currentLang); // Đặt ngôn ngữ ban đầu
}); 