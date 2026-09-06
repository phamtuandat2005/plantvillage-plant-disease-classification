# 🌿 PlantVillage - Phân Loại Bệnh Cây Trồng Bằng Thị Giác Máy Tính (Computer Vision)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Project%20Status-In%20Progress-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen.svg)](https://www.kaggle.com/datasets/emmarex/plantdisease)

Dự án ứng dụng các kỹ thuật **Thị giác máy tính (Computer Vision)** và **Học sâu (Deep Learning)** nhằm tự động phát hiện và phân loại các loại bệnh trên lá cây trồng từ hình ảnh, hỗ trợ bà con nông dân nhận biết bệnh sớm và nâng cao năng suất mùa màng.

---

## 📌 Báo cáo phân tích dữ liệu (EDA Report)

Báo cáo phân tích khám phá dữ liệu chi tiết đã được trích xuất tự động:
* 🌐 **Xem giao diện trực tiếp (HTMLPreview):** [Xem báo cáo tương tác tại đây](https://htmlpreview.github.io/?https://github.com/phamtuandat2005/plantvillage-plant-disease-classification/blob/main/analysis/plantvillage_data_analysis_report.html)
* 📑 **Báo cáo sau khi bật GitHub Pages:** `https://phamtuandat2005.github.io/plantvillage-plant-disease-classification/analysis/plantvillage_data_analysis_report.html`

---

## 📖 Mục tiêu đề tài

1. **Bài toán đặt ra:** Bệnh hại trên cây trồng thường lây lan nhanh. Việc chẩn đoán thủ công bằng mắt thường đòi hỏi chuyên môn cao, dễ nhầm lẫn và thường chỉ phát hiện khi bệnh đã nặng.
2. **Giải pháp đề xuất:** Xây dựng hệ thống học sâu có khả năng tiếp nhận ảnh chụp lá cây và phân loại chính xác tình trạng sức khỏe của cây (khỏe mạnh hoặc mắc bệnh cụ thể).
3. **Phạm vi:** Tập trung vào bài toán phân loại đa lớp (Multi-class classification) trên 38 nhóm bệnh/cây trồng thuộc tập dữ liệu chuẩn PlantVillage.

---

## 📊 Tổng quan tập dữ liệu (PlantVillage Dataset)

* **Tổng số mẫu:** `54,305` ảnh chụp màu (Color images).
* **Định dạng:** Ảnh RGB, kích thước chuẩn hóa `256 x 256` pixels.
* **Số lượng lớp (Classes):** `38` nhãn (gồm 14 loài cây trồng khác nhau).
* **Các loài cây trồng trong dữ liệu:**
  * 🍎 Táo (*Apple*)
  * 🫐 Việt quất (*Blueberry*)
  * 🍒 Anh đào (*Cherry*)
  * 🌽 Ngô (*Corn*)
  * 🍇 Nho (*Grape*)
  * 🍊 Cam (*Orange*)
  * 🍑 Đào (*Peach*)
  * 🫑 Ớt chuông (*Pepper bell*)
  * 🥔 Khoai tây (*Potato*)
  * 🍓 Dâu tây (*Strawberry* & *Raspberry*)
  * 🌱 Đậu nành (*Soybean*)
  * 🎃 Bí ngô (*Squash*)
  * 🍅 Cà chua (*Tomato*)

> [!NOTE]
> **Đặc điểm phân bố:**
> Dữ liệu có sự mất cân bằng đáng kể giữa các lớp:
> * Lớp nhiều mẫu nhất: `Orange___Haunglongbing_(Citrus_greening)` (5,507 ảnh), `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (5,357 ảnh).
> * Lớp ít mẫu nhất: `Potato___healthy` (152 ảnh), `Apple___Cedar_apple_rust` (275 ảnh).
> 
> *Chi tiết thống kê từng nhãn có thể tham khảo tại tệp [analysis/data_info.csv](analysis/data_info.csv).*

---

## 📁 Cấu trúc thư mục (Project Structure)

```text
plantvillage-plant-disease-classification/
│
├── _ipynb/                                  # Jupyter Notebooks thí nghiệm & phân tích
│   └── 01_plantvillage_data_analysis.ipynb  # Phân tích EDA dữ liệu ảnh & trích xuất metadata
│
├── analysis/                                # Kết quả phân tích dữ liệu chuyên sâu
│   ├── data_info.csv                        # Thống kê số lượng mẫu theo từng nhãn (38 lớp)
│   ├── image_metadata.csv                   # Metadata từng ảnh (kích thước, độ tương phản, độ sáng)
│   └── plantvillage_data_analysis_report.html # Báo cáo trực quan tương tác (Profiling Report)
│
├── docs/                                    # Tài liệu nghiên cứu & báo cáo đề tài
│   ├── _md/                                 # Tài liệu định dạng Markdown
│   ├── _pdf/                                # Tài liệu xuất bản PDF
│   └── _word/                               # Bản thảo nghiên cứu (.docx)
│
├── plantvillage_dataset/                    # Thư mục chứa dữ liệu ảnh gốc
│   └── color/                               # 38 thư mục con ứng với 38 nhãn phân loại
│
├── .gitignore                               # Cấu hình bỏ qua các file không cần commit Git
├── requirements.txt                         # Danh sách các thư viện Python cần thiết
└── README.md                                # Hướng dẫn dự án (File này)
```

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án

### 1. Yêu cầu hệ thống
* Python `3.10` trở lên (Khuyến nghị 3.10 hoặc 3.11).
* Khuyến khích có GPU (NVIDIA CUDA) để tăng tốc huấn luyện mô hình học sâu.

### 2. Clone mã nguồn
```bash
git clone https://github.com/phamtuandat2005/plantvillage-plant-disease-classification.git
cd plantvillage-plant-disease-classification
```

### 3. Khởi tạo môi trường ảo (Virtual Environment)
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows:
.venv\Scripts\activate

# Kích hoạt trên Linux / macOS:
source .venv/bin/activate
```

### 4. Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Chạy Notebook phân tích
Khởi chạy Jupyter Notebook hoặc mở bằng VS Code:
```bash
jupyter notebook _ipynb/01_plantvillage_data_analysis.ipynb
```

---

## 🧭 Lộ trình phát triển (Roadmap)

- [x] **Giai đoạn 1: Khám phá & Đánh giá dữ liệu (EDA)**
  - [x] Thống kê phân bố nhãn, kích thước ảnh, hệ màu, tỷ lệ khung hình.
  - [x] Phân tích phân bố độ sáng (*brightness*), độ tương phản (*contrast*) và tìm ảnh ngoại lai (*outliers*).
  - [x] Xuất báo cáo tự động dạng HTML profiling (`plantvillage_data_analysis_report.html`).
- [ ] **Giai đoạn 2: Tiền xử lý dữ liệu (Preprocessing & Pipeline)**
  - [ ] Phân chia tập dữ liệu (Train / Validation / Test) với kỹ thuật Stratified Split.
  - [ ] Xử lý mất cân bằng lớp (Class Imbalance) bằng Data Augmentation, Class Weights hoặc Oversampling.
  - [ ] Xây dựng luồng nạp dữ liệu tối ưu với `tf.data` (Prefetching, Caching, Batching).
- [ ] **Giai đoạn 3: Huấn luyện & Tối ưu mô hình (Modeling)**
  - [ ] Thử nghiệm Baseline CNN (kiến trúc tùy biến đơn giản).
  - [ ] Áp dụng Transfer Learning & Fine-tuning với các kiến trúc tiên tiến:
    - MobileNetV2 / MobileNetV3 (tối ưu cho thiết bị nhẹ)
    - ResNet50 / ResNet101
    - EfficientNet-B0/B2
  - [ ] Tối ưu hóa siêu tham số (Learning rate scheduling, Early stopping, Dropout).
- [ ] **Giai đoạn 4: Đánh giá mô hình (Evaluation)**
  - [ ] Đánh giá qua các chỉ số: Accuracy, Precision, Recall, Macro/Weighted F1-score.
  - [ ] Trực quan hóa ma trận nhầm lẫn (Confusion Matrix).
  - [ ] Giải thích mô hình với Grad-CAM (vùng ảnh kích hoạt quyết định phân loại).
- [ ] **Giai đoạn 5: Triển khai ứng dụng (Deployment)**
  - [ ] Xây dựng Web App tương tác nhận diện bệnh qua ảnh (Streamlit / Gradio / FastAPI).

---

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python
* **Học sâu & Thị giác máy tính:** TensorFlow / Keras, OpenCV, Pillow
* **Xử lý & Phân tích dữ liệu:** NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib
* **Báo cáo dữ liệu:** YData-Profiling (`fg-data-profiling`)

---

## 👤 Tác giả & Đóng góp

* **Tác giả:** [phamtuandat2005](https://github.com/phamtuandat2005)
* **Kho lưu trữ:** [plantvillage-plant-disease-classification](https://github.com/phamtuandat2005/plantvillage-plant-disease-classification)

