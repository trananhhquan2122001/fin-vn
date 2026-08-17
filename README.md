# 📈 FINEX VN TERMINAL

> **Hệ thống Phân tích & Định giá Doanh nghiệp Tự động hóa**  
*A modular, resilient, and data-driven terminal for Vietnamese equity valuation.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 💡 Giới Thiệu (About The Project)

**FINEX VN Terminal** là nền tảng web phân tích và định giá chứng khoán tự động tại thị trường Việt Nam. Dự án ra đời với mục tiêu mang lại công cụ chẩn đoán sức khỏe tài chính doanh nghiệp minh bạch, hỗ trợ nhà đầu tư cá nhân tiếp cận các mô hình định giá chuẩn mực (Piotroski, Altman Z-Score, DCF, Benjamin Graham) theo hướng hoàn toàn tự động.

Ứng dụng được thiết kế theo kiến trúc **Module Phân vùng linh hoạt (Step-Indexing)** giúp tối ưu hóa hiệu năng, chống sập giao diện và dễ dàng bảo trì/mở rộng.

---

## ✨ Tính Năng Nổi Bật (Key Features)

* **🟢 Tải Dữ Liệu Chống Sập 3 Tầng (Anti-Crash Fallback):** Tự động luôn chuyển giữa các máy chủ dữ liệu (`VCI`, `TCBS`, `MSN`) đảm bảo không bị ngắt kết nối.
* **🛡️ Chẩn Đoán Sức Khỏe Tài Chính:** Tính toán tự động chỉ số **Piotroski F-Score** (9 tiêu chí) và **Altman Z-Score** cảnh báo rủi ro doanh nghiệp.
* **📊 Định Giá Tự Động:** Tích hợp mô hình Chiết khấu Dòng tiền (DCF) và công thức Định giá Benjamin Graham cải tiến.
* **⚡ Giao Diện An Toàn (UI Safety Guard):** Duy trì khung hiển thị 100%, không bị trắng trang ngay cả khi kết nối mạng chập chờn.

---

## 🛠️ Cấu Trúc Mã Nguồn (Modular Architecture)

Mã nguồn `app.py` được quy chuẩn hóa bằng **Kỹ thuật Đánh số Phân vùng (Section Indexing)** để cộng đồng dễ dàng tra cứu và đóng góp:

| Phân Vùng | Tên Chức Năng | Mô Tả |
| :--- | :--- | :--- |
| `[SECTION 10]` | **Setup & Import** | Khai báo thư viện và cấu hình trang Streamlit |
| `[SECTION 20]` | **Data Engine (Fallback 3 Tầng)** | Tự động lấy Giá Realtime, Chỉ số Định giá & BCTC |
| `[SECTION 60]` | **Pure Math Algorithms** | Hàm tính điểm Piotroski F-Score & Altman Z-Score |
| `[SECTION 70]` | **Valuation Models** | Mô hình định giá DCF, Graham, P/E, P/B |
| `[SECTION 150]` | **Corporate Dashboard** | Phân tích bức tranh toàn cảnh doanh nghiệp |
| `[SECTION 200]` | **UI Header & Metrics** | Thanh tìm kiếm và 5 thẻ thông số nhanh |
| `[SECTION 250]` | **UI Tabs & Display** | Hiển thị chi tiết BCTC và Biểu đồ trực quan |

---

## 🚀 Hướng Dẫn Cài Đặt Cục Bộ (Local Installation)

Nếu bạn muốn chạy thử nghiệm dự án ở máy cá nhân:

1. **Clone repository về máy:**
   ```bash
   git clone [https://github.com/USERNAME/fin-vn.git](https://github.com/USERNAME/fin-vn.git)
   cd fin-vn
