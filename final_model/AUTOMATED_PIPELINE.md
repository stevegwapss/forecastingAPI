# Automated Retraining & Prediction Pipeline

**Updated:** October 27, 2025  
**Model:** Hybrid Transfer Learning with Adaptive Weighting  

---

## Overview

The automated pipeline handles:
1. ✅ **Monthly retraining** with latest data
2. ✅ **Automatic prediction generation** with adaptive weighting
3. ✅ **Email notifications** with results
4. ✅ **Model validation** and backup

---

## How It Works

### 1. Schedule
- **Trigger:** 1st of every month at 2:00 AM
- **Cron:** `0 2 1 * * ` 
- **Expected:** New month's data added to CSV by the 1st

### 2. Workflow Steps

#### Step 1: Check Data Availability
- Verifies that previous month's data exists in CSV
- Example: On Oct 1, expects Sept data
- If data missing: skips retrain, sends notification

#### Step 2: Validate Data Quality
- Checks for missing/negative values
- Detects potential outliers (>3 std dev)
- Ensures minimum 50 months of data
- If validation fails: aborts, sends error notification

#### Step 3: Backup Current Model
- Creates timestamped backup in `model_backups/`
- Preserves all `.keras` and `.json` files
- Allows rollback if needed

#### Step 4: Train New Model
- Runs `train_model.py` (Hybrid Transfer Learning)
- Pre-trains base LSTM on total_cases
- Fine-tunes per category with transfer learning
- Saves to `models/` directory:
  - `production_base_model.keras`
  - `production_category_*.keras` (one per category)
  - `production_model_config.json`

#### Step 5: Validate New Model
- Checks MAE threshold (default: 7.0)
- Reads from `production_model_config.json`
- If MAE too high: aborts, restores backup

#### Step 6: Generate Predictions ✨ NEW!
- Automatically runs `predict_next_month()` with new model
- Uses **adaptive weighting** based on volatility:
  - **STABLE** (<20%): 30% LSTM + 70% Baseline
  - **MODERATE** (20-50%): 50% LSTM + 50% Baseline  
  - **VOLATILE** (>50%): 70% LSTM + 30% Baseline
- Saves to `predictions/latest_prediction.json`
- Includes:
  - Predictions per category
  - Total predicted cases
  - Volatility percentage
  - Adaptive mode used
  - LSTM vs Baseline breakdown

#### Step 7: Send Success Notification
- Emails summary with:
  - Training duration
  - Model MAE
  - Total data months
  - **Prediction summary** (predicted total, volatility, mode)
  - Model type (Hybrid Transfer Learning with Adaptive Weighting)

---

## Model Details

### Training (Step 4)
**File:** `train_model.py`

```python
# Uses ALL available data
df_train = df_full.copy()

# Configuration
config = {
    'lookback': 6,
    'lstm_units': 32,
    'dropout_rate': 0.2,
    'dense_units': 16,
    'epochs': 100,
    'batch_size': 4,
    'patience': 20,
    'learning_rate': 0.001,
    'fine_tune_lr': 0.0005,
    'fine_tune_epochs': 50,
    'blend_ratio': 0.5  # Base ratio before adaptive adjustment
}
```

### Prediction (Step 6)
**File:** `predict.py`

```python
# Baseline: Last month in CSV
baseline_predictions[cat] = float(df[cat].iloc[-1])

# Adaptive weighting based on 3-month volatility
volatility = calculate_volatility(df, categories, lookback_months=3)
lstm_weight, baseline_weight, mode = get_adaptive_weights(volatility)

# Hybrid prediction
hybrid_predictions[cat] = (
    lstm_weight * transfer_predictions[cat] + 
    baseline_weight * baseline_predictions[cat]
)
```

**Example Output:**
```json
{
  "prediction_date": "2025-10-27T16:30:00",
  "predicting_for": "2025-09-01",
  "adaptive_weighting": {
    "enabled": true,
    "volatility": 44.6,
    "mode": "MODERATE",
    "lstm_weight": 0.5,
    "baseline_weight": 0.5
  },
  "predictions": {
    "by_category": {
      "Respiratory": 45,
      "Gastrointestinal": 32,
      ...
    },
    "total": 187
  }
}
```

---

## Environment Variables

### Required for Email Notifications
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
ALERT_EMAIL=recipient@example.com
```

### Optional Configuration
```bash
DATA_PATH=../data/monthly_data_8cat_no_covid.csv
MAE_THRESHOLD=7.0  # Abort if model MAE exceeds this
```

---

## Setup & Usage

### Manual Test (Recommended First)
```powershell
cd C:\xampp\htdocs\Uzi_Care\modeltraining\final_model
py test_automated_retrain.py
```

This runs the full pipeline without sending emails.

### Manual Run (Production)
```powershell
cd C:\xampp\htdocs\Uzi_Care\modeltraining\final_model
py automated_retrain.py
```

### Automated Schedule

#### Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. **Trigger:** Monthly, 1st day at 2:00 AM
4. **Action:** Start a program
   - Program: `py`
   - Arguments: `C:\xampp\htdocs\Uzi_Care\modeltraining\final_model\automated_retrain.py`
   - Start in: `C:\xampp\htdocs\Uzi_Care\modeltraining\final_model`
5. **Conditions:** Run only if computer is on, wake to run

#### Linux/Cloud (cron)
```bash
# Add to crontab
0 2 1 * * cd /path/to/modeltraining/final_model && python3 automated_retrain.py
```

---

## What Happens When New Data Arrives

### Example: September Data Added on October 1

1. **Before retrain:**
   - CSV has data through August 2025
   - Current model trained on Jan 2018 - Aug 2025
   - Baseline for predictions = August actuals

2. **September data added to CSV:**
   - New row: `2025-09-01, 45, 32, 28, ...`
   - CSV now has Jan 2018 - Sept 2025

3. **Automated retrain runs (Oct 1, 2:00 AM):**
   - ✅ Detects Sept data available
   - ✅ Validates data quality
   - ✅ Backs up old model
   - ✅ Trains new model on Jan 2018 - Sept 2025
   - ✅ Validates new model (MAE check)
   - ✅ **Generates predictions for October:**
     - Baseline = Sept actuals
     - Calculates volatility from June-Sept
     - Determines adaptive mode
     - Blends LSTM + Baseline
   - ✅ Saves `predictions/latest_prediction.json`
   - ✅ Sends email notification

4. **After retrain:**
   - Model now includes Sept in training
   - Predictions ready for October
   - API serves updated predictions
   - Baseline for next prediction = Sept actuals

---

## Baseline Behavior for New Months

### How Baseline is Computed

**Current Strategy:** Last month (single value)

```python
# In predict.py
baseline_predictions[cat] = float(df[cat].iloc[-1])
```

**For newly added month:**
- CSV ends at 2025-09 (Sept)
- Prediction for Oct uses Sept as baseline
- Baseline computed at runtime (not pre-stored)

**Alternative Strategy (if needed):**
```python
# 3-month average
baseline_predictions[cat] = float(df[cat].tail(3).mean())

# 3-month weighted (50%, 30%, 20%)
last_3 = df[cat].tail(3).values
baseline_predictions[cat] = float(
    last_3[-1]*0.5 + last_3[-2]*0.3 + last_3[-3]*0.2
)
```

---

## Files Generated

### Training Outputs
```
models/
├── production_base_model.keras (base LSTM)
├── production_category_Respiratory.keras
├── production_category_Gastrointestinal.keras
├── ... (one per category)
└── production_model_config.json (metadata)
```

### Prediction Outputs
```
predictions/
└── latest_prediction.json
```

### Backups
```
model_backups/
└── backup_20251001_020000/
    ├── production_base_model.keras
    ├── production_category_*.keras
    └── production_model_config.json
```

---

## Monitoring & Troubleshooting

### Check Last Run
```powershell
# View latest prediction
cat C:\xampp\htdocs\Uzi_Care\modeltraining\predictions\latest_prediction.json

# Check model config
cat C:\xampp\htdocs\Uzi_Care\modeltraining\final_model\models\production_model_config.json
```

### Common Issues

#### 1. "New month's data not yet available"
- **Cause:** CSV doesn't have expected month
- **Fix:** Add new month's data to CSV before retrain
- **Expected:** On Oct 1, Sept data should exist

#### 2. "Data quality issues detected"
- **Cause:** Missing values, negatives, or extreme outliers
- **Fix:** Clean the CSV data, validate inputs

#### 3. "Model validation failed. MAE too high"
- **Cause:** New model performs worse than threshold
- **Fix:** Check data quality, adjust MAE_THRESHOLD env var

#### 4. "Prediction generation failed"
- **Cause:** Error in predict.py or insufficient data
- **Fix:** Training still completes; run prediction manually:
  ```powershell
  cd C:\xampp\htdocs\Uzi_Care\modeltraining\final_model
  py predict.py
  ```

---

## Performance Metrics

### Validation Results (6-Month Rolling Test)
- **Average MAE:** 6.37 cases/category
- **R² Score:** 0.4433 (explains 44% of variance)
- **Volume Accuracy:** 82.9%

### Adaptive Weighting Impact
- **vs Fixed 50/50:** 12.6% better MAE
- **vs Pure Transfer:** 10.2% better MAE
- **vs Baseline only:** Competitive, more stable

---

## Next Steps

### Immediate
- [x] Auto-train implemented
- [x] Auto-predict implemented
- [x] Adaptive weighting integrated

### Optional Improvements
- [ ] Add API endpoint to trigger retrain manually
- [ ] Store prediction history (time series of predictions)
- [ ] Add dashboard to visualize retrain history
- [ ] Implement A/B testing for baseline strategies
- [ ] Add Slack/Discord notifications as alternative to email

---

## Contact

For issues or questions about the automated pipeline, check:
- `automated_retrain.py` - Main pipeline script
- `train_model.py` - Training logic
- `predict.py` - Prediction logic with adaptive weighting
- `FINAL_PROJECT_REPORT.md` - Full project documentation

**Last Updated:** October 27, 2025  
**Model Version:** 1.0.0 (Hybrid Transfer Learning with Adaptive Weighting)
