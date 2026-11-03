# 🌿 PlantVillage - Phân Loại Bệnh Cây Trồng Bằng Thị Giác Máy Tính (Computer Vision)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
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

* **Tổng số ảnh gốc:** `54,305` ảnh chụp màu (Color images).
* **Số ảnh sau khi làm sạch:** `54,284` ảnh (đã loại bỏ 21 ảnh trùng lặp tuyệt đối qua MD5 hash matching tại Notebook `02_data_cleaning.ipynb`).
* **Định dạng:** Ảnh RGB, kích thước gốc `256 x 256` pixels; kích thước chuẩn hóa cho mô hình VGG16: `224 x 224` pixels.
* **Số lượng lớp (Classes):** `38` nhãn (thuộc 14 loài cây trồng khác nhau).
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
> *Chi tiết thống kê từng nhãn có thể tham khảo tại tệp [analysis/data_info.csv](analysis/data_info.csv) và [outputs/vgg16/reports/class_distribution_splits.csv](outputs/vgg16/reports/class_distribution_splits.csv).*

---

## 📁 Cấu trúc thư mục (Project Structure)

```text
plantvillage-plant-disease-classification/
│
├── _ipynb/                                  # Jupyter Notebooks thí nghiệm & phân tích
│   ├── 01_plantvillage_data_analysis.ipynb  # Phân tích EDA dữ liệu ảnh & trích xuất metadata
│   ├── 02_data_cleaning.ipynb               # Làm sạch dữ liệu, lọc ảnh trùng & phân tích độ mờ/sáng
│   ├── 03_data_split_preprocessing.ipynb    # Phân chia Stratified Split (70/15/15) & mã hóa nhãn
│   ├── 04_cnn_VGG16.ipynb                   # Thử nghiệm sơ bộ kiến trúc VGG16
│   ├── 02_vgg16_experiments.ipynb          # Notebook thực nghiệm VGG16 hoàn chỉnh (Exp A, B, C & Grad-CAM)
│   └── 05_model_testing_and_comparison.ipynb # Kiểm thử & so sánh đa mô hình (.keras, .h5, .pth, 9 ảnh chẩn đoán)
│
├── analysis/                                # Kết quả phân tích dữ liệu chuyên sâu
│   ├── data_info.csv                        # Thống kê số lượng mẫu theo từng nhãn (38 lớp)
│   ├── image_metadata.csv                   # Metadata từng ảnh (kích thước, độ tương phản, độ sáng)
│   └── plantvillage_data_analysis_report.html # Báo cáo trực quan tương tác (Profiling Report)
│
├── configs/                                 # File cấu hình thực nghiệm
│   └── vgg16_config.json                    # Cấu hình siêu tham số, đường dẫn, aug và experiments
│
├── docs/                                    # Tài liệu nghiên cứu & báo cáo học thuật
│   └── _md/                                 # Báo cáo kỹ thuật chi tiết (vgg16_transfer_learning_report.md)
│
├── outputs/                                 # Toàn bộ artifacts và kết quả thực nghiệm
│   └── vgg16/
│       ├── checkpoints/                     # Checkpoints trọng số tốt nhất (.keras)
│       ├── histories/                       # Lịch sử huấn luyện (CSVLogger & metadata JSON)
│       ├── metrics/                         # Chỉ số đánh giá Test & Confusion Matrix (CSV, JSON)
│       ├── plots/                           # Biểu đồ Loss/Accuracy & Confusion Matrix Heatmap (PNG)
│       ├── gradcam/                         # Hình ảnh giải thích mô hình bằng Grad-CAM (3-panel)
│       └── reports/                         # Class mapping, phân phối mẫu và so sánh thực nghiệm
│
├── scripts/                                 # CLI entry points phục vụ tái lập nghiên cứu
│   ├── run_experiments.py                   # CLI Runner chạy toàn bộ 3 thực nghiệm và Grad-CAM
│   └── verify_pipeline.py                   # Script Smoke Test kiểm tra toàn bộ pipeline tự động
│
├── src/                                     # Core package mã nguồn chuẩn hóa module
│   ├── config.py                            # Cấu hình dự án (backward compatibility)
│   ├── data/
│   │   ├── dataset.py                       # Nạp dữ liệu, stratified split, class weight & tf.data
│   │   ├── preprocessing.py                 # Nạp ảnh RGB, resize 224x224 & vgg16.preprocess_input
│   │   └── augmentation.py                  # Pipeline Data Augmentation (RandomFlip, Rotation, Zoom...)
│   ├── models/                              # Kiến trúc mô hình mạng nơ-ron
│   │   └── vgg16_model.py                   # Kiến trúc VGG16 (GAP + Dropout 0.3 + Dense 38)
│   ├── training/                            # Huấn luyện và tối ưu hóa
│   │   ├── callbacks.py                     # Callbacks (ModelCheckpoint, EarlyStopping, ReduceLR, Timing)
│   │   ├── early_stopping_custom.py         # Lớp EarlyStopping tùy biến
│   │   └── train.py                         # Engine biên dịch và huấn luyện mô hình
│   ├── evaluation/                          # Đánh giá khoa học & giải thích mô hình
│   │   ├── evaluate.py                      # Đánh giá Test, Confusion Matrix, Classification Report
│   │   └── gradcam.py                       # Thuật toán Grad-CAM trên block5_conv3 (mẫu đúng & sai)
│   └── utils/                               # Tiện ích nghiên cứu
│       ├── config.py                        # Quản lý đường dẫn và nạp file cấu hình JSON
│       ├── reproducibility.py               # Cố định random seeds và trích xuất môi trường chạy
│       └── load_image.py                    # Tiện ích đọc ảnh
│
├── plantvillage_dataset/                    # Thư mục chứa dữ liệu ảnh gốc
│   ├── color/                               # 38 thư mục con ứng với 38 nhãn phân loại (54,305 ảnh)
│   └── csv/                                 # Metadata & Danh sách phân chia train/val/test
│       ├── cleaned_metadata.csv             # 54,284 mẫu sau làm sạch
│       ├── train.csv                        # 37,998 mẫu (70%)
│       ├── val.csv                          # 8,143 mẫu (15%)
│       └── test.csv                         # 8,143 mẫu (15%)
│
├── run_experiments.py                       # Root shortcut cho scripts/run_experiments.py
├── verify_pipeline.py                       # Root shortcut cho scripts/verify_pipeline.py
├── pyproject.toml                           # Định nghĩa package cho editable install (`pip install -e .`)
├── requirements.txt                         # Danh sách các thư viện Python phân nhóm rõ ràng
└── README.md                                # Hướng dẫn dự án (File này)
```

---

## 🔬 VGG16 Transfer Learning & Fine-tuning Pipeline

### 1. Kiến trúc mô hình VGG16 Tùy biến

Mô hình kế thừa trọng số tiền huấn luyện từ **ImageNet** (Simonyan & Zisserman, 2014) với cấu trúc classifier được thiết kế tối ưu cho bài toán nhận dạng bệnh lá cây:

```text
Input (224 x 224 x 3)
      │
      ▼
VGG16 Convolutional Backbone (13 Conv Layers trong 5 Blocks, MaxPooling)
      │  [Output: 7 x 7 x 512 feature map]
      ▼
GlobalAveragePooling2D
      │  [Output: 512-dim vector]
      ▼
Dropout (rate = 0.3)
      │
      ▼
Dense (38 units, activation = 'softmax')
```

**Thống kê tham số:**
* **Feature Extraction (Backbone đóng băng):**
  * Tổng tham số: `14,734,184`
  * Tham số huấn luyện (Trainable): `19,494` (chỉ tại Dense Head)
  * Tham số đóng băng (Non-trainable): `14,714,690`
* **Fine-tuning (Mở khóa Block 5: `block5_conv1`, `block5_conv2`, `block5_conv3`):**
  * Tham số huấn luyện (Trainable): `7,098,918`
  * Tham số đóng băng (Non-trainable): `7,635,266` (Blocks 1 đến 4)

> [!IMPORTANT]
> **Quy tắc tiền xử lý VGG16:**
> VGG16 yêu cầu chuẩn hóa bằng hàm `tf.keras.applications.vgg16.preprocess_input`: chuyển hệ màu RGB sang BGR và trừ giá trị trung bình kênh ImageNet `[103.939, 116.779, 123.68]`. Tuyệt đối **không áp dụng `Rescaling(1./255)`** trước VGG16 vì sẽ làm sai lệch phân phối đặc trưng đã học từ ImageNet.

### 2. Thiết kế các thực nghiệm khoa học

| Thực nghiệm | Kiến trúc & Trọng số | Tầng huấn luyện | Optimizer & LR | Augmentation | Callbacks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Experiment A (Feature Extraction)** | VGG16 ImageNet | Chỉ Classifier Head ($19.494$ params) | Adam, $\text{LR}=10^{-3}$ | Có | EarlyStopping, ReduceLR, ModelCheckpoint |
| **Experiment B (Fine-tuning)** | Checkpoint tốt nhất Exp A | Block 5 + Head ($7.098.918$ params) | Adam, $\text{LR}=10^{-5}$ | Có | EarlyStopping, ReduceLR, ModelCheckpoint |
| **Experiment C (Ablation Study)** | VGG16 ImageNet | Chỉ Classifier Head | Adam, $\text{LR}=10^{-3}$ | **Không** | EarlyStopping, ReduceLR, ModelCheckpoint |

### 3. Phân chia dữ liệu & Xử lý mất cân bằng lớp
* **Stratified Split (70/15/15):** Cố định seed 425; tỷ lệ giữa các lớp đồng nhất trên cả 3 tập; giao tập giữa các split bằng 0 (Zero Leakage).
* **Class Weighting:** Tính toán `class_weight` cân bằng trực tiếp từ tập Train để bù trừ mất cân bằng (từ lớp ít nhất 152 mẫu đến lớp nhiều nhất 5.507 mẫu).

### 4. Giải thích mô hình bằng Grad-CAM (Model Explainability)
* Sử dụng tầng tích chập cuối cùng `block5_conv3`.
* Lựa chọn có quy tắc cả **mẫu dự đoán đúng** và **mẫu dự đoán sai** trên tập Test.
* Xuất ảnh đối chiếu 3 khung hình: `[Ảnh gốc RGB] | [Bản đồ nhiệt Grad-CAM] | [Ảnh chồng Heatmap]` kèm nhãn thực tế, nhãn dự đoán và độ tin cậy (Confidence score).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thực Nghiệm

### 1. Khởi tạo môi trường ảo & Cài đặt thư viện

```bash
# Clone repository
git clone https://github.com/phamtuandat2005/plantvillage-plant-disease-classification.git
cd plantvillage-plant-disease-classification

# Tạo và kích hoạt môi trường ảo (Python 3.10 khuyến nghị)
python -m venv .env
.env\Scripts\activate       # Windows
# source .env/bin/activate  # Linux / macOS

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Chạy kiểm thử tự động (Smoke Test)

Kiểm tra toàn bộ pipeline (Data loader, Preprocessing, Model building, Forward pass, Callbacks, Đánh giá Test và Grad-CAM) trong chưa đầy 1 phút:

```bash
python verify_pipeline.py
```

### 3. Huấn luyện mô hình với Gemini CLI Studio (`train.py`)

Giao diện dòng lệnh tương tác hiện đại (TUI) phong cách **Gemini CLI / Claude Code**, hiển thị đo lường phần cứng, bảng màu sắc và menu chọn cấu hình từng bước:

```bash
# Khởi chạy giao diện tương tác (Interactive Studio Wizard):
python train.py

# Hoặc khởi chạy nhanh bằng cờ tham số (Headless / Script Mode):
# 1. Huấn luyện 2 giai đoạn chuẩn nghiên cứu (Stage 1 Warm-up -> Stage 2 Fine-tune Block 5):
python train.py --mode two_stage --epochs_stage1 15 --epochs_stage2 15 --batch_size 32

# 2. Huấn luyện Feature Extraction (chỉ huấn luyện Classifier Head):
python train.py --mode feature_extraction --epochs 20 --lr 1e-3

# 3. Huấn luyện Fine-Tuning tiếp từ checkpoint có sẵn:
python train.py --mode fine_tuning --resume outputs/vgg16/checkpoints/vgg16_feature_extraction_best.keras --epochs 15 --lr 1e-5

# 4. Kiểm tra nhanh (Quick Benchmark mode trên CPU):
python train.py --mode two_stage --quick --run_name test_run

# 5. Đánh giá riêng lẻ checkpoint và trực quan hóa Grad-CAM:
python train.py -i   # Chọn mục [6] trong menu
```

### 4. Chạy toàn bộ thực nghiệm đối chuẩn tự động (`run_experiments.py`)

Chạy tuần tự cả 3 cấu hình thực nghiệm (Exp A, Exp B, Exp C) để xuất bảng so sánh đối chuẩn khoa học:

```bash
# Chạy toàn bộ thực nghiệm đối chuẩn (Exp A, Exp B, Exp C):
python run_experiments.py --experiment all

# Chạy kiểm thử đối chuẩn nhanh:
python run_experiments.py --experiment all --quick
```

### 4. Chạy thực nghiệm tương tác qua Jupyter Notebook

Khởi chạy Notebook thực nghiệm khoa học:
```bash
# 1. Pipeline thực nghiệm VGG16 (Exp A, B, C & Grad-CAM)
jupyter notebook _ipynb/02_vgg16_experiments.ipynb

# 2. Kiểm thử và so sánh đa định dạng mô hình (.keras, .h5, .pth) & trực quan hóa chẩn đoán 9 ảnh
jupyter notebook _ipynb/05_model_testing_and_comparison.ipynb
```

---

## 🧭 Lộ trình phát triển (Roadmap)

- [x] **Giai đoạn 1: Khám phá & Đánh giá dữ liệu (EDA)**
  - [x] Thống kê phân bố nhãn, kích thước ảnh, hệ màu, tỷ lệ khung hình.
  - [x] Phân tích phân bố độ sáng (*brightness*), độ tương phản (*contrast*) và tìm ảnh ngoại lai (*outliers*).
  - [x] Xuất báo cáo tự động dạng HTML profiling (`plantvillage_data_analysis_report.html`).
- [x] **Giai đoạn 2: Tiền xử lý dữ liệu (Preprocessing & Pipeline)**
  - [x] Phân chia tập dữ liệu (Train / Validation / Test: 70/15/15) bằng Stratified Split, xác minh Zero Leakage.
  - [x] Xử lý mất cân bằng lớp (Class Imbalance) bằng Data Augmentation và balanced `class_weight`.
  - [x] Xây dựng luồng nạp dữ liệu tối ưu với `tf.data` (Prefetching, Batching, vgg16.preprocess_input).
- [x] **Giai đoạn 3: Huấn luyện & Tối ưu mô hình VGG16 (Modeling)**
  - [x] Triển khai VGG16 Feature Extraction (Frozen Backbone, GlobalAveragePooling2D, Dropout 0.3).
  - [x] Triển khai VGG16 Fine-tuning (Unfreeze Block 5 với Learning Rate nhỏ $10^{-5}$).
  - [x] Thực hiện Ablation Study về Data Augmentation (So sánh Không Augmentation vs Có Augmentation).
  - [x] Quản lý Checkpoints, EarlyStopping, ReduceLROnPlateau và CSVLogger.
- [x] **Giai đoạn 4: Đánh giá mô hình & Giải thích (Evaluation & Explainability)**
  - [x] Đánh giá trên tập Test độc lập qua các chỉ số: Accuracy, Precision, Recall, Macro/Weighted F1-score.
  - [x] Trực quan hóa và lưu trữ ma trận nhầm lẫn 38x38 (Confusion Matrix Heatmap và dữ liệu CSV).
  - [x] Giải thích mô hình với Grad-CAM trên tầng `block5_conv3` (phân tích cả mẫu đúng và mẫu sai).
- [ ] **Giai đoạn 5: Mở rộng kiến trúc & Triển khai ứng dụng (Deployment)**
  - [ ] So sánh đối chuẩn với MobileNetV3, ResNet50 và EfficientNet.
  - [ ] Xây dựng Web App tương tác nhận diện bệnh qua ảnh (Streamlit / Gradio / FastAPI).

---

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.10+
* **Học sâu & Thị giác máy tính:** TensorFlow 2.20 / Keras 3, OpenCV, Pillow
* **Xử lý & Phân tích dữ liệu:** NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib
* **Báo cáo dữ liệu:** YData-Profiling (`fg-data-profiling`)

---

## 👤 Tác giả & Đóng góp

* **Tác giả:** [phamtuandat2005](https://github.com/phamtuandat2005)
* **Kho lưu trữ:** [plantvillage-plant-disease-classification](https://github.com/phamtuandat2005/plantvillage-plant-disease-classification)
