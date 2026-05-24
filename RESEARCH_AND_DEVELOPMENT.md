# RESEARCH AND DEVELOPMENT - CD1 Finance

## 1. Tiến hóa kiến trúc hệ thống

### Kiến trúc cũ

```text
backend/
  app.py                 API lớn, chứa routing + inference + helper IO
  train_ann.py           train riêng, lặp get_dataset_path/eval_metrics/GRA
  train_lr.py            train riêng, lặp get_dataset_path/eval_metrics/GRA
  train_tree.py          train riêng, lặp get_dataset_path/eval_metrics/GRA
  train_xgb.py           train riêng, lặp get_dataset_path/eval_metrics/GRA
  preprocess_utils.py    preprocessing + indicator calculator
  train_wrapper.py       wrapper subprocess
```

Vấn đề chính của kiến trúc cũ:

- Logic đánh giá, tìm dataset, GRA và feature importance bị nhân bản ở nhiều file.
- API `app.py` vừa làm routing, vừa load model, vừa xử lý nghiệp vụ nên khó mở rộng.
- Huấn luyện từng model độc lập, chưa có meta-model hợp nhất xác suất vỡ nợ.
- Evaluation còn thiếu chỉ số phù hợp với dữ liệu mất cân bằng và dữ liệu tài chính trôi theo thời gian.

### Kiến trúc mới

```text
backend/
  api/
    server.py: Flask API mỏng, thread-safe bằng TRAIN_LOCK cho tác vụ train.

  core/
    config.py
      Cấu hình tập trung: đường dẫn, seed, device, UTF-8, logging.
    pipeline.py
      Abstract pipeline contract và dataclass kết quả huấn luyện.
    utils.py
      Dataset resolver, metric suite, PSI, save/load JSON, inference helper.

  services/
    preprocessing.py
      FinancialIndicatorCalculator và DataPreprocessor nâng cấp.
    feature_analysis.py
      GRA, tree-based importance, hybrid feature scoring và feature pruning.

  models/
    base_models.py
      ANN, Logistic Regression, LightGBM, XGBoost pipeline thống nhất.
    ensemble_model.py
      Heterogeneous Stacking Ensemble Classifier.

  scripts/
    train_runner.py
      Orchestrator duy nhất cho train từng model, train ensemble, train all.

  tools/
    Các tiện ích phát triển không thuộc runtime, ví dụ tạo sample dataset.

Các wrapper train cũ đã được loại bỏ để root backend chỉ còn entrypoint thật sự cần thiết.
```

## 2. Đặc tả triển khai kỹ thuật

### A. Dynamic Time-Series Hybrid Resampling: SMOTE-ENN

Trong mỗi expanding window:

```text
train = dữ liệu năm < t
test  = dữ liệu năm = t
X_train, y_train = tách feature/label
selected_features = HybridFeatureSelection(X_train, y_train)
X_res, y_res = SMOTEENN(X_train[selected_features], y_train)
model.fit(X_res, y_res)
```

SMOTE tạo mẫu tổng hợp cho lớp vỡ nợ thiểu số. Edited Nearest Neighbors sau đó loại bỏ các điểm biên gây nhiễu. Việc resampling chỉ chạy trong fold train, tuyệt đối không đụng tới fold test, vì vậy không gây rò rỉ thời gian.

File triển khai chính:

- `backend/models/base_models.py`: `_resample()`
- `backend/requirements.txt`: thêm `imbalanced-learn`

Fallback nghiên cứu:

- Nếu fold có quá ít mẫu ở một lớp, pipeline bỏ qua SMOTE-ENN cho fold đó để tránh tạo mẫu giả không đáng tin cậy.
- Nếu `imbalanced-learn` chưa cài, pipeline vẫn chạy với dữ liệu gốc và ghi warning.

### B. Embedded Hybrid Feature Selection Layer

Feature selection được thực hiện động trong từng fold, không thay đổi định nghĩa X1-X13.

Công thức điểm lai:

```text
GRA_j  = GreyRelationalCoefficient(X_j, y)
IMP_j  = ExtraTreesFeatureImportance(X_j, y)
RED_j  = max correlation giữa X_j và các biến còn lại

Score_j = 0.45 * norm(GRA_j)
        + 0.45 * norm(IMP_j)
        - 0.10 * RED_j
```

Quy trình:

```text
1. Tính GRA để nắm quan hệ tuyến tính/xấp xỉ hệ thống giữa chỉ tiêu tài chính và default.
2. Tính ExtraTrees importance để nắm quan hệ phi tuyến.
3. Trừ penalty cho biến có tương quan quá cao với biến khác.
4. Chọn biến có score tốt, giữ tối thiểu 6 feature để tránh pipeline quá nghèo thông tin.
5. Lưu bảng feature_influence_analysis_<model>.csv.
```

File triển khai chính:

- `backend/services/feature_analysis.py`
- `backend/models/base_models.py`: gọi `select_hybrid_features()` trong từng fold và trước final model.

### C. Heterogeneous Stacking Ensemble Classifier

Ensemble giải quyết xung đột giữa 4 model nền bằng cách học lại từ xác suất OOF:

```text
meta_X = [
  P_ann(default),
  P_lr(default),
  P_tree(default),
  P_xgb(default)
]
meta_y = default_final

meta_model = Calibrated Logistic Regression(meta_X, meta_y)
final_pd = meta_model.predict_proba(meta_X_new)[1]
```

Nguồn dữ liệu meta:

- `results/ann/predictions_with_probability.csv`
- `results/lr/predictions_with_probability.csv`
- `results/tree/predictions_with_probability.csv`
- `results/xgb/predictions_with_probability.csv`

Nếu chưa có OOF của model nền, ensemble tự gọi pipeline huấn luyện model nền trước. Với tập dữ liệu quá nhỏ hoặc số mẫu mỗi lớp không đủ cho calibration CV, hệ thống fallback về Logistic Regression cân bằng lớp.

File triển khai chính:

- `backend/models/ensemble_model.py`
- `backend/scripts/train_runner.py`

## 3. Khung đánh giá thực nghiệm và metric suite

### Time-Series Expanding Window

Thiết lập validation:

```text
Fold 1: train <= 2018, test = 2019
Fold 2: train <= 2019, test = 2020
Fold 3: train <= 2020, test = 2021
...
```

Nguyên tắc:

- Không shuffle theo thời gian.
- Feature selection và SMOTE-ENN chỉ fit trên train fold.
- Metric fold được lưu tại `metrics_by_year.csv`.
- Overall metric được tính trên toàn bộ OOF predictions, không dùng final model đã train full data để chấm lại lịch sử.

### Metric suite

Các chỉ số được tính trong `backend/core/utils.py`:

- **AUC-ROC**: đo năng lực xếp hạng xác suất default.
- **F1-Score**: cân bằng precision và recall.
- **G-Mean**: `sqrt(sensitivity * specificity)`, đặc biệt quan trọng khi dữ liệu default mất cân bằng.
- **Brier Score**: trung bình sai số bình phương giữa xác suất dự báo và nhãn thật; càng thấp càng tốt, phản ánh calibration.
- **PSI**: đo độ trôi phân phối feature giữa các expanding folds. PSI cao báo hiệu môi trường tài chính thay đổi mạnh, cần cẩn trọng khi diễn giải metric.

Quy ước PSI tham khảo:

```text
PSI < 0.10   : ổn định
0.10 - 0.25 : có drift trung bình
PSI > 0.25  : drift mạnh, cần kiểm tra robustness
```

## 4. API contract nâng cấp

### `POST /train`

Request:

```json
{
  "model_type": "xgb",
  "params": {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 160
  }
}
```

`model_type` hỗ trợ:

```text
ann, lr, tree, xgb, ensemble, all
```

Response:

```json
{
  "success": true,
  "model_type": "xgb",
  "metrics": {
    "AUC": 0.81,
    "F1": 0.62,
    "GMean": 0.73,
    "Brier": 0.18,
    "PSI": 0.07
  },
  "selected_features": ["X5", "X2", "X9"]
}
```

### `POST /predict`

Request:

```json
{
  "features": {
    "X1": 0.12,
    "X2": -0.03,
    "X3": 0.07,
    "X4": 0.20,
    "X5": 0.61
  }
}
```

Response mới:

```json
{
  "success": true,
  "data": {
    "ann": {"probability": 42.1},
    "lr": {"probability": 38.4},
    "tree": {"probability": 55.0},
    "xgb": {"probability": 47.8}
  },
  "ensemble": {
    "probability": 45.6,
    "base_models": ["ann", "lr", "tree", "xgb"],
    "confidence": 0.088
  },
  "input_features": {
    "X1": 0.12,
    "X2": -0.03
  }
}
```

### `POST /predict-from-file`

Response mới giữ `predictions` cá nhân và thêm `ensemble`:

```json
{
  "success": true,
  "data": {
    "maxYear": 2024,
    "nextYear": 2025,
    "companyCode": "ABC",
    "features": {"X1": 0.1},
    "predictions": {
      "ann": {"probability": 40.2},
      "lr": {"probability": 39.7},
      "tree": {"probability": 51.0},
      "xgb": {"probability": 44.5}
    },
    "ensemble": {
      "probability": 43.9,
      "base_models": ["ann", "lr", "tree", "xgb"],
      "confidence": 0.122
    }
  },
  "companies": ["ABC", "XYZ"]
}
```

### Frontend TypeScript contract

Đã thêm `frontend/src/types/api.ts` với:

- `PredictResponse`
- `PredictFromFileResponse`
- `EnsemblePrediction`
- `MetricSuite`

Axios client có thể map trực tiếp `response.data.ensemble.probability` làm xác suất vỡ nợ hợp nhất, đồng thời vẫn hiển thị từng model trong `response.data.data`.

## 5. Ghi chú vận hành

- Flask chạy `threaded=True`.
- Training endpoint được bảo vệ bằng `TRAIN_LOCK` để tránh hai job ghi cùng artifact.
- UTF-8 được bật trong `core/config.py` và `api/server.py`.
- Dockerfile hiện đọc `backend/requirements.txt`; dependency mới gồm `imbalanced-learn`, `xgboost`, `torch`, `openpyxl`.
- Lệnh train chuẩn hiện tại là `python -m scripts.train_runner <model_type>` khi đứng trong thư mục `backend`.

## 6. Rigorous Mathematical Audit, Model Upgrades (LightGBM), and Adaptive Macroeconomic Calibration (MDCL)

### 6.1. Temporal Leakage Audit

Audit mới nhất loại bỏ rủi ro leakage ở tầng preprocessing. Trước đây `DataPreprocessor.preprocess()` có thể fit `StandardScaler` trên toàn bộ dataframe trước khi expanding-window validation chạy. Nếu scaler học mean/variance từ các năm tương lai, fold quá khứ sẽ nhận thông tin phân phối tương lai:

```text
mu_global, sigma_global = fit(X_2015...X_2024)
X_2016_scaled = (X_2016 - mu_global) / sigma_global
```

Điều này vi phạm nguyên tắc temporal isolation vì `mu_global` và `sigma_global` chứa thông tin từ sau năm 2016.

Sau audit:

```text
services/preprocessing.py:
  - pd.to_numeric
  - KNNImputer
  - outlier clipping
  - không fit StandardScaler toàn cục
```

Scaling hiện nằm trong model pipeline của từng fold. Với fold dự báo năm `T`:

```text
train = {Year < T}
test  = {Year = T}

scaler.fit(X_train_resampled)
X_train_scaled = scaler.transform(X_train_resampled)
X_test_scaled  = scaler.transform(X_test)
```

Vì scaler chỉ fit trên `X_train_resampled`, validation year không tham gia vào bất kỳ thống kê chuẩn hóa nào. Đây là điều kiện cần để metric expanding-window có giá trị học thuật.

### 6.2. Dataset Schema Alignment

Dataset hiện tại là:

```text
backend/dataset/Processed_EWS_Final.csv
```

Schema gốc:

```text
Ticker, Year, ROA, ROE, Gross_Margin, Current_Ratio,
Debt_to_Assets, Cash_Ratio, Inventory_Turnover,
Receivables_Turnover, Target
```

Hệ thống tự chuẩn hóa schema:

```text
Ticker -> MaCP
Target -> default_final
STT tự tạo nếu chưa có
```

Feature inference không còn giả định X1-X13. Nếu không có X1-X13, hệ thống tự chọn các cột numeric hợp lệ sau khi loại:

```text
STT, MaCP, Year, default_final, split
```

### 6.3. Dynamic Feature Filtering At Prediction

Feature selection trong training có thể chọn subset động, ví dụ:

```text
ROA, Debt_to_Assets, ROE, Inventory_Turnover,
Receivables_Turnover, Current_Ratio
```

Do đó inference không được truyền toàn bộ cột thô theo thứ tự tùy ý. Mỗi model lưu:

```text
results/<model>/final_model/feature_schema.json
```

Tại prediction:

```text
schema = load_feature_schema(model)
X = incoming_data[schema]
model.predict_proba(X)
```

Nếu feature thiếu, hệ thống điền `0.0` theo đúng tên cột. Cách này đảm bảo orientation của ma trận inference khớp chính xác với ma trận lúc fit.

### 6.4. LightGBM Upgrade

Model identifier `tree` được giữ nguyên để không phá API và folder `results/tree`, nhưng estimator bên trong đã được nâng cấp:

```text
Decision Tree / Random Forest -> LightGBM LGBMClassifier
```

Lý do:

- Credit-risk tabular data thường skewed và phi tuyến.
- LightGBM dùng leaf-wise growth, học tương tác feature mạnh hơn tree đơn lẻ.
- Brier score thường tốt hơn nếu điều chỉnh regularization và class weight.

Cấu hình mặc định:

```text
n_estimators = 300
learning_rate = 0.03
num_leaves = 15
max_depth = 4
min_child_samples = 25
subsample = 0.85
colsample_bytree = 0.85
reg_alpha = 0.1
reg_lambda = 1.0
class_weight = balanced
```

### 6.5. Logistic Regression Calibration

Logistic Regression hiện được wrap bằng:

```text
CalibratedClassifierCV(LogisticRegression, method="sigmoid", cv=3)
```

Mục tiêu là làm xác suất mượt hơn và hạn chế extreme confidence. Đây là bước quan trọng vì hệ thống không chỉ phân loại default/non-default, mà còn hiển thị Default Probability (%).

### 6.6. MDCL: Macroeconomic Drift Calibration Layer

MDCL là lớp hiệu chỉnh quyết định dựa trên drift. Hệ thống lấy PSI trung bình từ các base models:

```text
PSI_mean = mean(PSI_ann, PSI_lr, PSI_tree, PSI_xgb)
```

Ngưỡng quyết định thích nghi:

```text
tau_adaptive = 0.5 - 0.3 * min(PSI_mean, 0.5)
```

Diễn giải:

- Khi PSI thấp, dữ liệu ổn định, `tau_adaptive` gần 0.5.
- Khi PSI cao, baseline tài chính dịch chuyển, hệ thống giảm ngưỡng để nhạy hơn với rủi ro.
- Nếu `PSI_mean > 0.15`, `mdcl_active = true`.

Prediction payload hiện trả thêm:

```json
{
  "ensemble": {
    "probability": 12.34,
    "psi_mean": 0.041,
    "adaptive_threshold": 0.4877,
    "adjusted_risk_level": "LOW RISK",
    "mdcl_active": false
  }
}
```

Quy tắc:

```text
if ensemble_probability >= tau_adaptive:
    adjusted_risk_level = HIGH RISK
else:
    adjusted_risk_level = LOW RISK
```

Lớp này không thay đổi raw probability, mà bổ sung quyết định có ngữ cảnh drift để phục vụ báo cáo học thuật và phân tích rủi ro trong môi trường kinh tế biến động.
