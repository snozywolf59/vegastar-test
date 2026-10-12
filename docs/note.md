# Hạn chế, hiệu năng và hướng mở rộng

## 1. Hạn chế đã biết

### Độ tin cậy

- Agent hiện chạy bên trong coroutine của request SSE. Nếu client ngắt kết nối hoặc process API restart, run chưa có `run_id`, event log và worker độc lập để tiếp tục/reconnect.
- Lỗi tool được phát qua event `error` nhưng agent vẫn có thể nhận chuỗi lỗi như một `ToolMessage`; cần quy định rõ lỗi nào retry và lỗi nào kết thúc run.
- Chưa có điều chỉnh timeout riêng cho LLM, embedding; chưa có circuit breaker hoặc retry có backoff.

### Chất lượng dữ liệu và AI

- Summary là nén mất mát; chi tiết không được giữ trong summary sẽ không thể tìm bằng vector dù message gốc vẫn còn trong DB.
- Vector retrieval phụ thuộc câu query do agent tạo, threshold và embedding model; tham chiếu như “tàu lúc nãy” có thể có similarity thấp.
- Tuyến AIS nối các điểm theo thời gian bằng đường thẳng; không suy ra tuyến hàng hải thực và có thể tạo đoạn nhảy do dữ liệu lỗi/GPS drift.

### Bảo mật và vận hành

- Chưa có authentication/authorization; biết UUID có thể đọc hoặc xóa hội thoại.
- Chưa có rate limit, quota theo người dùng, audit log.

## 2. Độ trễ và chi phí ước tính

### 2.1. Điều kiện đo

Các số liệu dưới đây được đo thủ công với **một client**, chạy trên cùng máy deploy backend và database. Vì vậy, đây là kết quả tham khảo cho môi trường hiện tại, chưa phải SLA và chưa phản ánh tình huống nhiều người dùng đồng thời. Dữ liệu thu thập từ 24 câu trả lời.

- **TTFT (Time To First Token):** thời gian từ lúc gửi request đến khi frontend nhận token đầu tiên.
- **Full response:** thời gian từ lúc gửi request đến khi nhận xong toàn bộ câu trả lời.
- **Tool time:** thời gian thực thi hàm tool và truy vấn database, không bao gồm thời gian LLM quyết định gọi tool hoặc tổng hợp kết quả.

### 2.2. Kết quả test

Thời gian trung bình theo từng tool:

| Tool                            | Thời gian trung bình |
| ------------------------------- | -------------------: |
| `search_vessels`                |              11,0 ms |
| `get_vessel_route`              |              53,7 ms |
| `get_vessel_dark_gaps`          |             10.02 ms |
| `get_vessel_position_near_time` |              11.6 ms |
| `get_vessel_details`            |             11.23 ms |
| `get_vessel_ownership`          |                  9ms |
| `search_conversation_memory`    |                  4ms |

Trung bình của bốn request:

- TTFT khoảng **2,41 giây**.
- Thời gian hoàn thành khoảng **3,65 giây**.
- Riêng ba request có tool call: TTFT khoảng **2,82 giây**, hoàn thành khoảng **4,37 giây**.

### 2.3. Chi phí token

Model đang cấu hình là `gemini-3.5-flash-lite`. Theo bảng giá Gemini Developer API tại thời điểm cập nhật tài liệu:

- Input: **0,30 USD / 1 triệu token**.
- Output, bao gồm thinking token: **2,50 USD / 1 triệu token**.

Công thức:

```text
chi phí request = input_tokens / 1.000.000 × 0,30 USD
                + output_tokens / 1.000.000 × 2,50 USD
```

Số token input sử dụng trải đều từ 3000 --> 11000 token. Lượng token output từ 200-->800 token.
Có thể ước tính lượng input token trung bình \sim 8000 token/request và lượng output token trung bình \sim 600 token/request. Vậy trung bình tốn khoảng 0.0039 USD/request.
Nếu tỷ lệ bốn loại request trong thực tế giống mẫu test, chi phí LLM ước tính là:

| Số request |        Chi phí ước tính |
| ---------: | ----------------------: |
|      1.000 |     khoảng **3,90 USD** |
|     10.000 |    khoảng **39,00 USD** |
|    100.000 |   khoảng **390,00 USD** |
|  1.000.000 | khoảng **3.900,00 USD** |

Đây chỉ là chi phí token của model chat, chưa tính chi phí về hạ tầng lưu trữ.

## 3. Mở rộng đến vài chục triệu điểm AIS/ngày

Khi nhận 20–50 triệu điểm AIS mỗi ngày, bảng `ais_positions` sẽ tăng rất nhanh. Nếu tiếp tục lưu tất cả dữ liệu trong một bảng duy nhất, việc thêm dữ liệu, tạo index và truy vấn hành trình sẽ chậm dần. Có thể mở rộng theo các bước sau.

### chia bảng theo thời gian

Nên partition bảng `ais_positions` theo ngày hoặc theo tuần dựa trên `event_ts`.

Ví dụ, dữ liệu ngày 01/10 và ngày 02/10 nằm ở hai partition khác nhau. Khi mà hỏi hành trình trong ngày 02/10, PostgreSQL chỉ cần đọc partition của ngày đó thay vì quét toàn bộ dữ liệu.

### Nạp dữ liệu theo batch

Gom nhiều dòng thành một batch và nạp dữ liệu vào, có thể theo workflow sau:

1. Nhận file csv AIS.
2. Kiểm tra các trường bắt buộc như `vessel_id`, `event_ts`, `lat`, `lon`.
3. Loại bỏ dòng lỗi hoặc trùng lặp.
4. Ghi cả batch vào đúng partition.
5. Ghi log số dòng thành công và thất bại.

Việc nạp dữ liệu nên chạy bằng process hoặc worker riêng

### Theo dõi hệ thống

Các chỉ số nên theo dõi gồm:

- số điểm AIS được ghi mỗi phút;
- số dòng lỗi hoặc trùng lặp khi import;
- thời gian trung bình và chậm nhất của từng tool;
- dung lượng bảng và index theo ngày;
- CPU, RAM, disk I/O và số connection PostgreSQL.