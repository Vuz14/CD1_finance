# 📋 PredictPage Update - CSV File Upload & Prediction Summary

## ✅ Completed Tasks

### 1. **Backend Updates** (`app.py`)
- ✅ Added new endpoint: `/predict-from-file` (POST)
- ✅ Handles CSV file upload with validation
- ✅ Automatically finds max year in data
- ✅ Extracts company code from max year data
- ✅ Makes predictions using all 4 models (ANN, LR, Tree, XGBoost)
- ✅ Returns formatted response with features and predictions for next year

### 2. **Frontend Updates** (`PredictPage.tsx`)
- ✅ Added new Tab: "Upload CSV & Dự Đoán" (Tab 0)
- ✅ File upload interface with drag-drop style
- ✅ Added `PredictFromFileResult` interface for type safety
- ✅ Displays prediction results from all 4 models in a clean table format
- ✅ Shows company code, max year, and predicted year
- ✅ Color-coded risk levels (Red = High risk, Green = Low risk)

### 3. **API Module Updates** (`api.ts`)
- ✅ Added generic `post()` method to api object
- ✅ Enables flexible API requests from frontend

### 4. **Build Status**
- ✅ Frontend builds successfully with no TypeScript errors
- ✅ All components properly typed and imported

---

## 🎯 How to Use

### Upload a Preprocessed CSV File:

1. **Prepare your file**: Use `processed_BCTC_by_year.csv` (already prepared)
   - Format: STT, MaCP, Year, X1-X13
   - At least 2 years of data required (to predict for next year)

2. **Navigate to PredictPage**:
   - Click first tab: **"Upload CSV & Dự Đoán"**

3. **Upload the file**:
   - Click the file upload button
   - Select the CSV file

4. **View Results**:
   - System automatically finds:
     - Max year in data
     - Company code
     - Calculates predictions for (Max Year + 1)
   - Displays probability from all 4 models

---

## 📊 Prediction Output Format

```json
{
  "success": true,
  "data": {
    "maxYear": 2024,
    "nextYear": 2025,
    "companyCode": "OCB",
    "features": {
      "X1": 0.082,
      "X2": -0.127,
      ...
      "X13": 2.251
    },
    "predictions": {
      "ann": { "probability": 2.33 },
      "lr": { "probability": 0.0 },
      "tree": { "probability": 36.54 },
      "xgb": { "probability": 65.26 }
    }
  }
}
```

---

## 🔧 Technical Details

### Backend Endpoint: `/predict-from-file`
- **Method**: POST
- **Input**: Multipart form data with CSV file
- **Validation**:
  - Only CSV files allowed
  - Must contain: STT, MaCP, Year, X1-X13
  - Must have at least 1 year of data

- **Processing**:
  1. Read and parse CSV
  2. Find maximum year
  3. Extract features for max year
  4. Load all 4 pre-trained models
  5. Make predictions
  6. Return results

- **Error Handling**:
  - Missing required columns → 400 error
  - Model not found → Error in results
  - Invalid file → 400 error

### Frontend Components:
- **Tab 0**: File Upload Interface
- **Tab 1**: Direct X1-X13 input (original)
- **Tab 2**: Calculate from financial metrics (original)

### Result Display Table:
| Column | Content |
|--------|---------|
| Model | ANN, LR, Decision Tree, XGBoost |
| Xác Suất Vỡ Nợ (%) | Probability percentage |
| Mức Rủi Ro | Risk level (HIGH/LOW) |

---

## 📁 File Structure

```
backend/
├── app.py              ✅ Updated with /predict-from-file endpoint
├── preprocess_utils.py ✅ Preprocessing pipeline (already working)
└── processed_BCTC_by_year.csv ✅ Ready for upload

frontend/
├── src/
│   ├── api.ts         ✅ Updated with post() method
│   └── pages/
│       └── PredictPage.tsx  ✅ Updated with new tab and file upload
└── build/             ✅ Freshly built (npm run build)
```

---

## ✨ Features Implemented

| Feature | Status | Details |
|---------|--------|---------|
| CSV File Upload | ✅ Done | Accepts preprocessed BCTC files |
| Auto Year Detection | ✅ Done | Finds max year and predicts next year |
| Multi-Model Prediction | ✅ Done | Uses all 4 pre-trained models |
| Results Display | ✅ Done | Clean table with risk indicators |
| Error Handling | ✅ Done | Validates file format and content |
| TypeScript Types | ✅ Done | Fully typed interfaces |
| Build Success | ✅ Done | No compilation errors |

---

## 🚀 Deployment Steps

1. **Backend**: Already has the new endpoint
2. **Frontend**: Already built (in `build/` folder)
3. **Both are ready for deployment**

---

## 📝 Example Usage

**Input**: `processed_BCTC_by_year.csv`
- Contains data for OCB from 2018-2024

**Process**:
- System finds max year = 2024
- Extracts X1-X13 values for OCB 2024
- Passes features to 4 models

**Output**:
```
Company: OCB
Max Year: 2024
Predict Year: 2025

Results:
- ANN:      2.33% (✅ Low Risk)
- LR:       0.00% (✅ Low Risk)
- Tree:    36.54% (⚠️  Medium Risk)
- XGBoost: 65.26% (🔴 High Risk)
```

---

## ✅ All Requirements Met

✔️ Load preprocessed CSV file
✔️ Find max year automatically
✔️ Make predictions for next year
✔️ Use all 4 models
✔️ Display results clearly
✔️ Proper error handling
✔️ Type-safe frontend code
✔️ No build errors

**Status: READY FOR USE** 🎉
