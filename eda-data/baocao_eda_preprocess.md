# Báo cáo: Phân tích Dữ liệu Khám phá (EDA) và Tiền Xử lý (MovieLens 1M & Amazon-Book)

## 1. Giới thiệu
Báo cáo này trình bày tóm tắt quá trình phân tích dữ liệu khám phá (EDA), kiểm tra tính toàn vẹn và các quyết định tiền xử lý kỹ thuật đối với hai tập dữ liệu benchmark: **MovieLens 1M** và **Amazon-Book**. Mục tiêu của các bước xử lý này là chuẩn bị dữ liệu tuân thủ nghiêm ngặt các yêu cầu đầu vào của mô hình **Convolutional Neural Networks for Collaborative Filtering (ConvNCF)**.

---

## 2. Tập dữ liệu MovieLens 1M

### 2.1. Đặc điểm Dữ liệu Tổng quan
| Đặc trưng | Thông số kỹ thuật |
|---|---|
| Nguồn dữ liệu | GroupLens Research |
| Unique Users | 6.040 |
| Unique Items (Movies) | 3.883 |
| Tổng số tương tác | 1.000.209 |
| Mức độ thưa thớt (Sparsity) | 95,74% |
| Tương tác trung bình / User | 165,6 |
| Tương tác trung bình / Item | 269,9 |
| Loại phản hồi gốc | Explicit (Ratings: 1-5 sao) |
| Loại phản hồi sau xử lý | Implicit (Tương tác: 1) |
### 2.2. Phân phối Phản hồi (Trước Tiền Xử Lý)
Dữ liệu gốc cung cấp phản hồi rõ ràng (explicit feedback) dưới dạng điểm đánh giá (ratings). Quan sát phân phối cho thấy sự lệch chuẩn về phía các đánh giá tích cực (phần lớn tập trung ở mức 4 sao: 34,9% và 3 sao: 26,1%). 
**Lưu ý:** Sự phân phối này đã được điều chỉnh trong giai đoạn tiền xử lý cuối cùng (xem Mục 2.6) để phù hợp với kiến trúc của ConvNCF.

![Phân phối Rating - MovieLens 1M](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_rating_distribution.png)

### 2.3. Phân phối tương tác


````carousel
![Số ratings mỗi người dùng](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_ratings_per_user.png)
<!-- slide -->
![Số ratings mỗi phim](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_ratings_per_movie.png)
````

### 2.4. Top 10 phim được đánh giá nhiều nhất
![Top 10 phim](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_top10_movies.png)

### 2.5. Thông tin nhân khẩu học


````carousel
![Phân phối giới tính](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_gender_distribution.png)
<!-- slide -->
![Phân phối nhóm tuổi](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ml_age_distribution.png)
````

### 2.6. Quy trình Tiền xử lý và Chuyển đổi Dữ liệu

Dữ liệu MovieLens đã trải qua các bước chuẩn hóa sau:
1. **Remapping ID:** Chuẩn hóa toàn bộ `UserID` và `MovieID` về không gian số nguyên liên tục $\in [0, N-1]$ để tương thích với ma trận nhúng (embedding matrix).
2. **Chuyển đổi sang Implicit Feedback:** Mô hình ConvNCF tối ưu hóa hàm mục tiêu **Bayesian Personalized Ranking (BPR)**, đo lường sự khác biệt thứ hạng giữa mẫu dương và mẫu âm, do đó không sử dụng giá trị rating cụ thể. Mọi tương tác $\geq 1$ sao đều được chuyển đổi thành biến chỉ báo nhị phân (giá trị bằng 1).
3. **Phân chia Train/Test:** 
   - **Chiến lược:** Stratified Random Split theo `user_id` với tỷ lệ 80/20.
   - **Cơ sở lý luận:** Phân tầng (Stratified) đảm bảo độ đo đánh giá tính khách quan bằng cách duy trì dữ liệu lịch sử cho mọi người dùng ở cả hai tập, qua đó loại trừ triệt để **bài toán bắt đầu lạnh (cold-start problem)** trong không gian test.
   - **Quy mô:** Tập huấn luyện (800.167 mẫu), Tập kiểm thử (200.042 mẫu).

### 2.7. Xác minh Tính Toàn vẹn (Sau Xử lý)

| Tiêu chí Kiểm soát Chất lượng | Trạng thái / Kết quả |
|---|---|
| Giá trị khuyết thiếu (Missing values) | ✅ Vượt qua (0) |
| Định dạng Implicit Feedback | ✅ Vượt qua (Toàn bộ ratings = 1) |
| Mẫu ngoại lai (Outliers/Invalid IDs) | ✅ Vượt qua (0) |
| Bảo toàn số lượng bản ghi sau split | ✅ Khớp (1.000.209) |
| Cold-start users/items trong tập Test | ✅ Vượt qua (0) |
---

## 3. Tập dữ liệu Amazon-Book

### 3.1. Đặc điểm Dữ liệu Tổng quan
| Đặc trưng | Thông số kỹ thuật |
|---|---|
| Nguồn dữ liệu | Amazon Product Data (Julian McAuley) |
| Unique Users | 52.643 |
| Unique Items (Books) | 91.599 |
| Tổng số tương tác | 2.984.108 |
| Mức độ thưa thớt (Sparsity) | 99,94% |
| Tương tác trung bình / User | 56,7 |
| Tương tác trung bình / Item | 32,6 |
| Loại phản hồi | Implicit |
### 3.2. Phân phối tương tác

````carousel
![Tương tác mỗi người dùng](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ab_interactions_per_user.png)
<!-- slide -->
![Tương tác mỗi sản phẩm](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ab_interactions_per_item.png)
````

### 3.3. Cơ cấu Phân chia Phục vụ Mô hình hóa
Tập dữ liệu đã được nhóm nghiên cứu chuẩn bị sẵn ở cấu hình chia Train/Test chặt chẽ phục vụ cho tác vụ đánh giá Recommendation Systems. Hệ thống giữ nguyên cấu trúc này:
- **Tập Train:** 2.380.730 tương tác (79,78%). Định dạng: `User_ID Item_ID_1 Item_ID_2...`
- **Tập Test:** 603.378 tương tác (20,22%).

![Tỷ lệ Train/Test](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/ab_train_test_split.png)

### 3.4. Xác minh Tính Toàn vẹn


Dữ liệu đầu vào đã được xác thực tính đúng đắn và độc lập:
| Tiêu chí Kiểm soát Chất lượng | Trạng thái / Kết quả |
|---|---|
| Tính nhất quán của ID Map | ✅ Khớp (100% ID nằm trong user/item list) |
| Rò rỉ dữ liệu (Data Leakage) | ✅ Không ghi nhận sự giao thoa tương tác giữa Train/Test |
| Rủi ro Cold-start | ✅ Zero cold-start entities |
| Trùng lặp sự kiện (Duplications) | ✅ 0 (Đã kiểm chứng độc lập trên từng tệp) |
---

## 4. Bàn luận & So sánh Đánh giá (Comparative Analysis)

![So sánh MovieLens vs Amazon-Book](file:///Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My%20Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/eda_plots/comparison_ml_vs_ab.png)

| Tiêu chuẩn Đánh giá | MovieLens 1M | Amazon-Book |
|---|---|---|
| **Kích thước Dữ liệu** | Trung bình (1 triệu) | Lớn (Gần 3 triệu) |
| **Độ thưa thớt (Sparsity)** | 95,74% | **99,94%** |
| **Bản chất của Phản hồi** | Implicit (Đã chuyển đổi từ Explicit 1-5 sao) | Implicit thuần túy (Hành vi Mua/Xem) |
| **Quy mô Danh mục (Items)** | Hẹp (3.883 phim) | Cực rộng (91.599 sách) |

**Phân tích kỹ thuật:**
Thử nghiệm trên cả hai tập dữ liệu cho phép đánh giá toàn diện tính mạnh mẽ của kiến trúc hệ thống. MovieLens đóng vai trò là cơ sở để kiểm chứng khả năng trích xuất đặc trưng chung (nhờ mật độ tương tác tương đối tốt: 95.74%). Trong khi đó, Amazon-Book đại diện cho môi trường khắc nghiệt trong thực tế với độ thưa thớt lên đến gần 100% (99.94%) và không gian tìm kiếm (Items) lớn hơn gấp 23 lần, đặt ra thách thức lớn đối với bài toán cold-start cục bộ nội hàm và khả năng biểu diễn của các nhúng (embeddings) tiềm ẩn trong Convolutional NCF.

---

## 5. Tổng kết
- Cả hai bộ cơ sở dữ liệu đều đáp ứng các tiêu chuẩn nghiêm ngặt về tính toàn vẹn (integrity) cho việc huấn luyện Deep Learning.
- Quá trình chuyển đổi cấu trúc đánh giá MovieLens sang Implicit Feedback đã hoàn tất, đảm bảo hệ thống có thể tối ưu hiệu quả trên cùng một khung Loss Function.
- Dữ liệu hoàn toàn sẵn sàng cho vòng đời huấn luyện, điều chỉnh siêu tham số và đánh giá các độ đo truy hồi thông tin (Hit Ratio, NDCG).
