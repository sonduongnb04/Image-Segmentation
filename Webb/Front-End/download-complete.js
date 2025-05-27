// Ghi chú: File này xử lý logic cho trang thông báo tải về hoàn tất (download-complete.html)
document.addEventListener('DOMContentLoaded', function () {
    // Lấy các phần tử DOM cần thiết
    const vnLangBtn = document.getElementById('vn-lang'); // Nút chọn tiếng Việt
    const enLangBtn = document.getElementById('en-lang'); // Nút chọn tiếng Anh
    const completeMessage = document.getElementById('complete-message'); // Dòng thông báo
    const continueBtn = document.getElementById('continue-btn'); // Nút "Tiếp tục"
    const copyrightText = document.getElementById('copyright-text'); // Dòng copyright

    // Cập nhật copyright year cho nhất quán
    const currentYear = new Date().getFullYear();

    // Định nghĩa các chuỗi dịch
    let currentLang = localStorage.getItem('language') || 'vi';
    const translations = {
        vi: { /* Bản dịch tiếng Việt */
            title: 'Tải về hoàn tất',
            header: 'Phân đoạn ảnh giao thông trực tuyến', // Giữ header nhất quán
            message: 'Tập tin của bạn đã sẵn sàng',
            continue: 'Tiếp tục',
            copyright: `© ${currentYear} Phân đoạn ảnh giao thông trực tuyến. Dự án nghiên cứu.`
        },
        en: { /* Bản dịch tiếng Anh */
            title: 'Download Complete',
            header: 'Online Traffic Image Segmentation', // Keep header consistent
            message: 'Your file is ready',
            continue: 'Continue',
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
        if (completeMessage) completeMessage.textContent = translations[lang].message;
        if (continueBtn) continueBtn.textContent = translations[lang].continue;
        if (copyrightText) copyrightText.textContent = translations[lang].copyright;
    }

    // --- Gán sự kiện --- 

    // Sự kiện click cho nút chọn ngôn ngữ
    if (vnLangBtn) vnLangBtn.addEventListener('click', () => setLanguage('vi'));
    if (enLangBtn) enLangBtn.addEventListener('click', () => setLanguage('en'));

    // Nút "Tiếp tục" đã có sẵn href="index.html" nên không cần thêm sự kiện click ở đây
    // để chuyển trang.

    // --- Khởi tạo --- 
    setLanguage(currentLang); // Đặt ngôn ngữ ban đầu khi tải trang
}); 