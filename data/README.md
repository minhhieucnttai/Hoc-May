# Data Directory

## Dataset: Global Disaster Response 2018-2024

### Mô tả
- **Tên file:** `global_disaster_response_2018_2024.csv`
- **Quy mô:** ~50,000 bản ghi
- **Thời gian:** 2018 - 2024
- **Biến mục tiêu:** `recovery_days`

### Cách tải dữ liệu

Do file dữ liệu lớn, bạn có thể:

1. **Sử dụng dữ liệu mẫu tự động tạo** (được tích hợp trong app.py)

2. **Tải từ các nguồn công khai:**
   - EM-DAT: https://public.emdat.be/
   - World Bank Open Data: https://data.worldbank.org/

### Cấu trúc dữ liệu

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| date | datetime | Ngày xảy ra thảm họa |
| country | string | Quốc gia |
| disaster_type | string | Loại thảm họa |
| severity_index | float | Chỉ số nghiêm trọng (1-10) |
| casualties | int | Số thương vong |
| economic_loss_usd | float | Thiệt hại kinh tế (USD) |
| response_time_hours | float | Thời gian phản ứng (giờ) |
| aid_amount_usd | float | Số tiền viện trợ (USD) |
| response_efficiency_score | float | Điểm hiệu quả (0-1) |
| latitude | float | Vĩ độ |
| longitude | float | Kinh độ |
| recovery_days | float | Số ngày phục hồi (TARGET) |

### Lưu ý
- Nếu không có file dữ liệu, chương trình sẽ tự động tạo dữ liệu mẫu
- Dữ liệu mẫu mô phỏng các đặc điểm thống kê của dataset thực
