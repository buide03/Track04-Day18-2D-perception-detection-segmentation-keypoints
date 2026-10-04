# Báo Cáo Thí Nghiệm Bonus — Lab 18 (2D Perception)

**Học viên thực hiện**: bui_dinh_de  
**Dataset**: `tiger-pose` (12 keypoints)  
**Môi trường thực nghiệm**: Tesla T4 GPU (Google Colab), `torch==2.11.0+cu130`, `ultralytics==8.4.171`

---

## 1. Thí Nghiệm 4C: Tập Val Lật Gương & So Sánh Hai Model (10 Điểm Bonus)

### 1.1. Bảng số liệu thực nghiệm (`RESULT_4C`)

Từ lần chạy thực tế trên máy, kết quả đánh giá giữa hai mô hình:

| Mô hình | Pose mAP50-95 (Val gốc) | Pose mAP50-95 (Val lật gương) | Biến thiên mAP |
|:---|:---:|:---:|:---:|
| **Model FLIP_IDX Giải Phẫu** (`pose_model`) | **0.457** | **0.439** | $-3.9\%$ (Ổn định) |
| **Model FLIP_IDX Đồng Nhất** (`m_id`) | **0.417** | **0.298** | **$-28.6\%$ (Sụp đổ)** |

---

### 1.2. Phân tích: Metric nào đã che giấu lỗi `flip_idx`?

1. **`Box mAP` hoàn toàn "mù" trước lỗi nhãn keypoint**:
   - Bounding box chỉ bao quanh hình chữ nhật ngoại tiếp cơ thể con hổ $(x_1, y_1, x_2, y_2)$ mà không chứa thông tin về cấu trúc xương khớp bên trong.
   - Dù con hổ quay sang trái hay phải, bounding box vẫn khoanh trúng cơ thể con hổ, do đó `Box mAP50-95` luôn giữ mức cao $(\approx 0.930)$ và không thể phát hiện ra việc hoán đổi nhãn chi.

2. **`Pose mAP` trên tập val gốc bị đánh lừa bởi Data Bias (Thiên lệch dữ liệu)**:
   - Thống kê hướng quay ở Mục 4A chỉ ra: Tập train có $210/210$ con hổ quay phải ($100\%$), tập val gốc có $53/53$ con hổ quay phải ($100\%$).
   - Khi `flip_idx` giữ nguyên đồng nhất `[0..11]`, mô hình chỉ cần "học vẹt" quy luật: *Chân hướng về phía camera luôn là `right_*`, chân ở phía xa là `left_*`*.
   - Vì tập val gốc không có bất kỳ con hổ nào quay sang trái, quy luật học vẹt này không hề bị trừng phạt $\rightarrow$ `Pose mAP` trên val gốc vẫn đạt $0.417$ (rất gần với $0.457$ của model chuẩn giải phẫu).

3. **Tập val lật gương bóc trần sự thật**:
   - Khi tạo tập `tiger-pose-mirror` (lật ngang ảnh và đổi nhãn theo đúng giải phẫu học, giả lập hổ quay sang trái):
   - Model `flip_idx` đồng nhất dự đoán chân hướng về camera là `right_*` (nhưng thực tế sinh học là chân trái `left_*`). Toàn bộ 4 cặp chân bị gán nhãn chéo nhau $\rightarrow$ `Pose mAP50-95` lập tức sụp đổ từ **$0.417$ xuống $0.298$** (giảm tới $28.6\%$).
   - Trong khi đó, Model `flip_idx` giải phẫu do được học cơ chế hoán đổi đối xứng nên giữ vững hiệu năng trên cả 2 hướng ($0.457 \rightarrow 0.439$).

### 1.3. Bài học thiết kế tập kiểm thử (Validation Design)
Một tập kiểm thử tốt phải phản ánh toàn diện phân phối trong đời thực, không được sao chép nguyên vẹn các thiên lệch (spurious correlations / biases) của tập huấn luyện. Nếu tập huấn luyện bị lệch hướng (như camera chỉ quay một chiều), bắt buộc phải chủ động tạo các tập stress-test (như lật gương, đổi góc chiếu) để kiểm tra tính tổng quát hóa thực sự của mô hình.

---

## 2. Báo Cáo Bài Tập Về Nhà: Đo Pipeline & Latency

Bảng đo đạc thời gian chi tiết giữa 2 head trên Tesla T4 GPU (trung bình 30 lần suy luận):

| Cấu hình | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Số candidate boxes |
|---|:---:|:---:|:---:|:---:|
| **One-to-many + NMS** (conf = 0.25) | 1.73 | 9.24 | **1.19** | 5 |
| **One-to-many + NMS** (conf = 0.001) | 1.76 | 8.91 | **1.32** | 203 |
| **One-to-one NMS-free** (conf = 0.25) | 1.73 | 9.43 | **0.43** | 5 |
| **One-to-one NMS-free** (conf = 0.001) | 1.79 | 9.68 | **0.45** | 204 |

**Kết luận kỹ thuật**:
- Ở mức confidence thấp (lúc đánh giá mAP COCO), head One-to-one NMS-free giúp tiết kiệm **66% thời gian hậu xử lý** ($1.32 \text{ ms} \rightarrow 0.45 \text{ ms}$).
- Kiến trúc NMS-free loại bỏ hoàn toàn nút thắt cổ chai tuần tự của giải thuật Greedy NMS, cho phép triển khai end-to-end tensor operations tối ưu trên các thiết bị biên (Edge AI, NPU, CPU).
