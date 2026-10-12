# Nghiên cứu và lựa chọn công nghệ

Tài liệu này ghi lại quyết định kỹ thuật.
Gồm các phần:
1. Tiêu chí lựa chọn công nghệ
2. Lựa chọn công nghệ
3. So sánh các chiến lược
4. Các bước thực hiện
5. Các biến cấu hình

## 1. Tiêu chí lựa chọn công nghệ

Có một số tiêu chí lựa chọn như sau:

- LLM có hỗ trợ tool call có cấu trúc và có thể streaming token.
- Xử lý tốt câu hỏi tiếng Việt và hội thoại nhiều lượt.
- Có embedding model để dễ vận hành.
- Các lựa chọn về LLM, database cần rẻ và dễ triển khai.
- Dễ triển khai bằng Docker, ít thành phần vận hành trong giai đoạn đầu.

## 2. Lựa chọn công  nghê hiện tại

| Thành phần      | Lựa chọn                  | Lý do chọn                                                                                               | Trade-off                                                                            |
| --------------- | ------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| LLM             | Gemini                    | Streaming, tool calling, hỗ trợ tiếng Việt; miễn phí, có embedding.                                      | Phụ thuộc quota, giá và độ ổn định của API bên ngoài.                                |
| Vector DB       | PostgreSQL + pgvector     | Dùng chung transaction và khóa ngoại với session, không cần vận hành vector DB riêng, có fuzzy matching. | Không tối ưu bằng hệ chuyên dụng khi corpus vector cực lớn hoặc phân tán nhiều vùng. |
| Spatial DB      | PostgreSQL + PostGIS      | Truy vấn điểm AIS, khoảng cách, tuyến đường và GeoJSON ngay trong DB.                                    | Cần partition/index/vacuum đúng cách khi dữ liệu lớn.                                |
| Agent framework | LangChain Core            | Chuẩn hóa message, tool schema, model binding và async streaming; không áp đặt graph phức tạp.           | Vòng lặp agent, persistence và retry vẫn phải tự triển khai.                         |
| API             | FastAPI + SSE             | Async, validation bằng Pydantic, tài liệu OpenAPI; SSE đơn giản cho luồng server → browser.              | Endpoint hiện gắn vòng đời agent với kết nối HTTP.                                   |
| UI              | HTML/JavaScript + Leaflet | Nhẹ, đủ để kiểm thử stream, Markdown và GeoJSON.                                                         | Chưa phải frontend production, chưa có state management và reconnect bền vững.       |

Các cấu hình được điều chỉnh qua file `.env`.

## 3. So sánh chiến lược hội thoại

| Chiến lược               | Cách hoạt động                                         | Ưu điểm                                                                                  | Hạn chế                                                           | Phù hợp                             |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------- |
| Cửa sổ trượt             | Chỉ gửi `N` lượt gần nhất                              | Đơn giản, dữ liệu gần đây chính xác                                                      | Quên các thông tin cũ                                             | Hội thoại ngắn, hỏi đáp độc lập     |
| Tóm tắt                  | Tạo tóm tắt tin nhắn cũ, xong đưa vào prompt  | Token thấp hơn, giữ được mạch tổng quát                                                      | Có mất mát thông tin; summary dài dần, loãng thông tin            | Hội thoại tuyến tính, ít đổi chủ đề |
| Kết hợp truy xuất vector | Embed lịch sử/chunk rồi tìm đoạn gần nghĩa với câu hỏi | Lấy lại chi tiết cũ theo nhu cầu, không gửi toàn bộ lịch sử trò chuyện, tính ổn định cao | Có thể bỏ sót tham chiếu mơ hồ; thêm latency và chi phí embedding | Hội thoại dài, nhiều chủ đề         |

Hệ thống dùng kết hợp có truy xuất chủ động:

1. Chỉ các lượt gần nhất trong `CHAT_CONTEXT_TURNS` được gửi nguyên văn cho LLM.
2. Khi lịch sử vượt cửa sổ, phần cũ chưa xử lý được tóm tắt thành một memory chunk.
3. Tóm tắt được embedding và lưu trong `session_memories`, kèm khoảng `sequence_start`–`sequence_end`.
4. Agent được cấp tool `search_conversation_memory`. Agent chỉ gọi tool này khi chi tiết cần thiết không còn trong cửa sổ gần.
5. Tìm kiếm bị giới hạn theo `session_id`, `embedding_model`, `MEMORY_TOP_K` và `MEMORY_MIN_SIMILARITY`, nên các cuộc hội thoại không lẫn dữ liệu.

Cách làm trên giúp giảm lượng token input, đồng thời tránh đưa thông tin cũ không liên quan và cho phép LLM quyết định khi nào cần nhớ lại. Cửa sổ trượt vẫn giữ nguyên văn các trao đổi mới nhất, summary/vector xử lý trí nhớ dài hạn.

Ngoài ra cũng cần thiết kế các tool SQL giúp truy xuất thông tin. Về phần này sẽ có các phương pháp chính sau:
| Phương pháp              | Chính xác                                                                               | An toàn                                                  | Chi phí token | Thời gian       | Độ phức tạp | Phù hợp với                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------- | --------------- | ----------- | ----------------------------------------------------------------------------- |
| Predefined SQL           | Cao với nghiệp vụ đã định nghĩa và kiểm thử                                             | Cao nếu dùng parameterized query và kiểm soát quyền      | Thấp          | Thấp            | Thấp        | Các nghiệp vụ phổ biến, truy vấn có cấu trúc rõ ràng                          |
| Text-to-SQL              | Phụ thuộc schema, prompt và độ phức tạp câu hỏi                                         | Trung bình–thấp nếu thiếu cơ chế kiểm soát               | Trung bình    | Trung bình      | Trung bình  | Câu hỏi tự nhiên đa dạng, cần truy vấn linh hoạt                              |
| SQL Agent                | Có thể cải thiện qua nhiều vòng kiểm tra và sửa lỗi, nhưng không đảm bảo đúng ngữ nghĩa | Rủi ro cao hơn nếu không giới hạn quyền và vòng thực thi | Cao           | Cao             | Cao         | Truy vấn phức tạp cần thực thi, đánh giá và điều chỉnh nhiều bước             |
| Hybrid SQL + Text-to-SQL | Cao nếu định tuyến đúng và kiểm tra kết quả                                             | Cao nếu giới hạn Text-to-SQL và kiểm soát quyền database | Trung bình    | Thấp–trung bình | Cao         | AI Agent khai thác dữ liệu đa dạng, cần cân bằng độ tin cậy và tính linh hoạt |

Với các yêu cầu của bài toán, tôi chọn phương pháp predefined sql để thiết kế các công cụ giúp agent quan sát thông tin. Vì với các tình huống cụ thể trong yêu cầu, ta hoàn toàn có thể thiết kế các lệnh sql để lấy thông tin, đảm bảo hoạt động ổn định và an toàn. Ngoài ra với text-to-sql hay sql agent, sẽ tốn thêm token và các phương pháp để đảm bảo câu lệnh sử dụng là chính xác.

Về obserbility, hiện tại đang sử dụng thêm bảng trong PosgreSQL để lưu về các thông tin về input, output, tool call và thời gian. Trong xây dựng sản phẩm thực tế có thể dùng Langfuse để tracking và quản lý các prompt dễ hơn.

## 4. Các bước thực thi

Đầu tiên cần thiết kế các bảng để lưu thông tin từ csv và đánh các index tương ứng. Ngoài các bảng lưu thông tin cơ bản từ csv, thiết kế thêm các bảng `messages`, `sessions`, `session_memories` để lưu trữ các thông tin hội thoại. Trong đó `messages` sẽ lưu các tin nhắn (của `user`, `assistant`, `tool`); `sessions` lưu thông tin chung về các cuộc hội thoại; `session_memories` lưu các thông tin đặc trưng của hội thoại (các bản tóm tắt, chi phí sử dụng,...). Ta sẽ cần đánh index ở các trường thông tin được tìm kiếm nhiều, có thể kể đến:
- Thông tin tàu: Tên tàu (cần thêm index để fuzzy match), mmsi,...
- Thông tin về vị trí địa lý
- Thông tin về thời gian

Bước thứ 2, ta cần nạp dữ liệu từ csv vào. Ở đây cần chú ý do dữ liệu đầu vào là thông tin thô từ csv, nên sử dụng python để tạo script tải. Sở dĩ dùng python, vì ta có thể thực hiện biến đổi, chuẩn hóa dữ liệu trước khi đưa vào lưu trữ.

Tiếp theo xây dựng phía backend với FastAPI + LangChain để phát triển nhanh về mặt xử lý thông tin và đưa ra câu trả lời. Phía frontend dùng html để sử dụng nhanh.
Về mặt xử lý dữ liệu, cần thiết kế các tool và prompt để agent có thông tin mà trả lời. Dựa trên các tình huống được cung cấp

Cuối cùng containize lại và dùng docker compose để sử dụng.

## 5. Cấu hình

| Biến                          | Vai trò                                   | Tác động khi tăng                                                            |
| ----------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| `CHAT_CONTEXT_TURNS`          | Số lượt user gần nhất được giữ nguyên văn | Tăng độ chính xác ngữ cảnh gần, đồng thời tăng input token                   |
| `AGENT_MAX_TOOL_ROUNDS`       | Số vòng LLM → tool tối đa                 | Giải được câu hỏi phức tạp hơn nhưng tăng latency/chi phí và rủi ro loop     |
| `MEMORY_TOP_K`                | Số memory trả về tối đa                   | Tăng recall nhưng tăng token và nhiễu                                        |
| `MEMORY_MIN_SIMILARITY`       | Ngưỡng cosine similarity                  | Tăng precision nhưng có thể bỏ sót memory                                    |
| `GEMINI_EMBEDDING_DIMENSIONS` | Số chiều vector                           | Có thể tăng chất lượng nhưng tăng dung lượng/index; phải khớp index pgvector |

Việc chọn giá trị nên dựa trên tập câu hỏi đánh giá:

- follow-up gần
- nhắc lại chi tiết rất cũ, đổi chủ đề
- tên tàu sai chính tả và câu hỏi cần nhiều tool.
