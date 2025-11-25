"""
AUTOMATED MONTHLY RETRAINING SCRIPT
====================================
Runs on the 1st of every month to:
1. Retrain Hybrid Transfer Learning model with latest data
2. Generate predictions using adaptive weighting
3. Send email notifications with results

The model uses:
- Transfer Learning: Pre-trained LSTM fine-tuned per category
- Baseline: Last month's actual cases
- Adaptive Weighting: Dynamically adjusts LSTM/Baseline ratio based on volatility
  - STABLE (<20%): 30% LSTM + 70% Baseline
  - MODERATE (20-50%): 50% LSTM + 50% Baseline
  - VOLATILE (>50%): 70% LSTM + 30% Baseline

Schedule: 0 2 1 * * (1st of month at 2:00 AM)
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from illness_categories import ILLNESS_CATEGORIES

def log_message(message, level="INFO"):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def send_notification(subject, message, success=True):
    """Send email notification (configure your SMTP settings)"""
    try:
        # Get email config from environment variables
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        sender_email = os.getenv('SENDER_EMAIL', '')
        sender_password = os.getenv('SENDER_PASSWORD', '')
        recipient_email = os.getenv('ALERT_EMAIL', '')
        
        if not all([sender_email, sender_password, recipient_email]):
            log_message("Email credentials not configured, skipping notification", "WARN")
            return
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"[Uzi Care] {subject}"
        
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: {'green' if success else 'red'};">
                    {'✅' if success else '❌'} {subject}
                </h2>
                <p>{message}</p>
                <hr>
                <p style="color: gray; font-size: 12px;">
                    Uzi Care Automated Retraining System<br>
                    Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        log_message("Notification email sent successfully")
    except Exception as e:
        log_message(f"Failed to send notification: {str(e)}", "ERROR")

def check_new_data_available(data_path):
    """Check if new month's data is available"""
    try:
        df = pd.read_csv(data_path)
        df['month'] = pd.to_datetime(df['month'])
        
        last_data_month = df['month'].max()
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Check if we have last month's data
        # (On Oct 1, we should have Sept data)
        expected_month = pd.Timestamp(current_month) - pd.DateOffset(months=1)
        
        log_message(f"Last data month: {last_data_month.strftime('%Y-%m')}")
        log_message(f"Expected month: {expected_month.strftime('%Y-%m')}")
        
        if last_data_month >= expected_month:
            return True, len(df)
        else:
            return False, len(df)
    except Exception as e:
        log_message(f"Error checking data availability: {str(e)}", "ERROR")
        return False, 0

def validate_data_quality(data_path):
    """Validate data quality before training"""
    try:
        df = pd.read_csv(data_path)
        
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
        
        # Check for outliers (values > 3 std from mean)
        for cat in ILLNESS_CATEGORIES:
            if cat in df.columns:
                mean = df[cat].mean()
                std = df[cat].std()
                outliers = df[df[cat] > mean + 3*std]
                if len(outliers) > 0:
                    issues.append(f"Potential outliers in {cat}: {len(outliers)} months")
        
        # Check minimum data points
        if len(df) < 50:
            issues.append(f"Insufficient data: only {len(df)} months (need 50+)")
        
        if issues:
            log_message("Data quality issues detected:", "WARN")
            for issue in issues:
                log_message(f"  - {issue}", "WARN")
            return False, issues
        
        log_message("Data quality validation passed ✅")
        return True, []
    except Exception as e:
        log_message(f"Error validating data: {str(e)}", "ERROR")
        return False, [str(e)]

def run_training():
    """Execute the training script"""
    log_message("Starting model training...")
    
    try:
        # Import and run training
        import train_model
        
        log_message("Training completed successfully ✅")
        return True
    except Exception as e:
        log_message(f"Training failed: {str(e)}", "ERROR")
        return False

def validate_model_performance(models_dir='models'):
    """Validate the newly trained model"""
    try:
        log_message("Validating model performance...")
        
        # Check if model files exist
        config_path = os.path.join(models_dir, 'production_model_config.json')
        if not os.path.exists(config_path):
            log_message("Model config not found", "ERROR")
            return False, None
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check performance metrics (if available)
        mae = config.get('training_mae', None)
        
        if mae is None:
            log_message("MAE not found in config, assuming OK", "WARN")
            return True, None
        
        # Check MAE threshold
        mae_threshold = float(os.getenv('MAE_THRESHOLD', '7.0'))
        
        log_message(f"Model MAE: {mae:.2f}")
        log_message(f"MAE Threshold: {mae_threshold:.2f}")
        
        if mae > mae_threshold:
            log_message(f"MAE {mae:.2f} exceeds threshold {mae_threshold:.2f}", "ERROR")
            return False, mae
        
        log_message("Model performance validation passed ✅")
        return True, mae
    except Exception as e:
        log_message(f"Error validating model: {str(e)}", "ERROR")
        return False, None

def backup_old_model(models_dir='models', backup_dir='model_backups'):
    """Backup previous model before replacing"""
    try:
        if not os.path.exists(models_dir):
            log_message("No existing models to backup")
            return True
        
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Copy all model files
        for filename in os.listdir(models_dir):
            if filename.endswith(('.keras', '.json')):
                src = os.path.join(models_dir, filename)
                dst = os.path.join(backup_path, filename)
                import shutil
                shutil.copy2(src, dst)
        
        log_message(f"Previous model backed up to {backup_path} ✅")
        return True
    except Exception as e:
        log_message(f"Failed to backup model: {str(e)}", "WARN")
        return False

def main():
    """Main retraining workflow"""
    log_message("=" * 80)
    log_message("AUTOMATED MONTHLY RETRAINING - STARTED")
    log_message("=" * 80)
    
    start_time = datetime.now()
    
    # Configuration
    data_path = os.getenv('DATA_PATH', '../data/monthly_data_8cat_no_covid.csv')
    models_dir = 'models'
    
    # Step 1: Check if new data is available
    log_message("Step 1: Checking data availability...")
    data_available, num_months = check_new_data_available(data_path)
    
    if not data_available:
        message = f"New month's data not yet available. Current data: {num_months} months. Skipping retrain."
        log_message(message, "INFO")
        send_notification("Retrain Skipped - Data Not Ready", message, success=True)
        return
    
    log_message(f"New data available! Total months: {num_months} ✅")
    
    # Step 2: Validate data quality
    log_message("Step 2: Validating data quality...")
    quality_ok, issues = validate_data_quality(data_path)
    
    if not quality_ok:
        message = f"Data quality issues detected:\n" + "\n".join(issues)
        log_message(message, "ERROR")
        send_notification("Retrain Failed - Data Quality Issues", message, success=False)
        sys.exit(1)
    
    # Step 3: Backup old model
    log_message("Step 3: Backing up previous model...")
    backup_old_model(models_dir)
    
    # Step 4: Run training
    log_message("Step 4: Training new model...")
    training_success = run_training()
    
    if not training_success:
        message = "Model training failed. Check logs for details."
        log_message(message, "ERROR")
        send_notification("Retrain Failed - Training Error", message, success=False)
        sys.exit(1)
    
    # Step 5: Validate new model
    log_message("Step 5: Validating new model...")
    validation_ok, mae = validate_model_performance(models_dir)
    
    if not validation_ok:
        message = f"Model validation failed. MAE: {mae if mae else 'unknown'}"
        log_message(message, "ERROR")
        send_notification("Retrain Failed - Validation Error", message, success=False)
        sys.exit(1)
    
    # Step 6: Generate predictions with newly trained model
    log_message("Step 6: Generating predictions with new model...")
    prediction_success = False
    prediction_result = None
    
    try:
        from predict import predict_next_month
        
        # Run prediction with adaptive weighting (default)
        prediction_result = predict_next_month(
            historical_data_path=data_path,
            use_adaptive_weights=True
        )
        
        prediction_success = True
        
        predicting_for = prediction_result.get('predicting_for', 'Unknown')
        predicted_total = prediction_result.get('predictions', {}).get('total', 'N/A')
        volatility = prediction_result.get('adaptive_weighting', {}).get('volatility', 'N/A')
        mode = prediction_result.get('adaptive_weighting', {}).get('mode', 'N/A')
        
        log_message(f"✅ Prediction generated successfully")
        log_message(f"   Predicting for: {predicting_for}")
        log_message(f"   Predicted total: {predicted_total} cases")
        log_message(f"   Volatility: {volatility}% ({mode} mode)")
        
    except Exception as e:
        log_message(f"Warning: Prediction generation failed: {str(e)}", "WARN")
        log_message("Training completed but predictions not generated", "WARN")
    
    # Step 7: Success!
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log_message("=" * 80)
    log_message("AUTOMATED MONTHLY RETRAINING - COMPLETED SUCCESSFULLY ✅")
    log_message("=" * 80)
    log_message(f"Duration: {duration:.2f} seconds")
    log_message(f"Model MAE: {mae if mae else 'N/A'}")
    log_message(f"Total data months: {num_months}")
    
    if prediction_success and prediction_result:
        predicting_for = prediction_result.get('predicting_for', 'Unknown')
        predicted_total = prediction_result.get('predictions', {}).get('total', 'N/A')
        volatility = prediction_result.get('adaptive_weighting', {}).get('volatility', 'N/A')
        mode = prediction_result.get('adaptive_weighting', {}).get('mode', 'N/A')
        
        message = f"""
    Model retraining and prediction completed successfully!
    
    📊 Training Summary:
    - Duration: {duration:.2f} seconds
    - Model MAE: {mae if mae else 'N/A'}
    - Total data months: {num_months}
    
    🔮 Prediction Summary:
    - Predicting for: {predicting_for}
    - Predicted total: {predicted_total} cases
    - Market volatility: {volatility}% ({mode} mode)
    - Model type: Hybrid Transfer Learning with Adaptive Weighting
    
    The new model is now deployed and predictions are ready!
    """
    else:
        message = f"""
    Model retraining completed successfully!
    
    Duration: {duration:.2f} seconds
    Model MAE: {mae if mae else 'N/A'}
    Total data months: {num_months}
    
    The new model is now deployed and ready for predictions.
    (Note: Automatic prediction generation failed - run manually if needed)
    """
    
    send_notification("Retrain Successful ✅", message, success=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message = f"Unexpected error in automated retraining: {str(e)}"
        log_message(error_message, "ERROR")
        send_notification("Retrain Failed - Unexpected Error", error_message, success=False)
        sys.exit(1)
