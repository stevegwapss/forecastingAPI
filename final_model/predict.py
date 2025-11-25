"""
Production Prediction Module
=============================
Use trained Hybrid Transfer Learning model to predict next month's cases
"""

import json
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from illness_categories import get_all_categories

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'monthly_data_8cat_no_covid.csv')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

def load_production_model():
    """Load all trained models and configuration"""
    
    # Load config
    config_path = os.path.join(MODELS_DIR, 'production_model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load base model (not used for prediction, but good to have)
    base_model_path = os.path.join(MODELS_DIR, 'production_base_model.keras')
    base_model = keras.models.load_model(base_model_path)
    
    # Load category models
    categories = config['categories']
    category_models = {}
    
    for cat in categories:
        safe_cat_name = cat.replace('_', '-').replace('/', '-')
        model_path = os.path.join(MODELS_DIR, f'production_category_{safe_cat_name}.keras')
        category_models[cat] = keras.models.load_model(model_path)
    
    return config, category_models

def calculate_category_volatility(df, categories, lookback_months=12):
    """
    Calculate volatility for EACH category individually over lookback period
    
    Args:
        df: DataFrame with historical data
        categories: List of category names
        lookback_months: Number of months to analyze (default: 12)
        
    Returns:
        dict: {category: volatility_percentage}
    """
    cat_volatilities = {}
    
    for cat in categories:
        recent_data = df[cat].tail(lookback_months + 1).values
        
        if len(recent_data) < 2:
            cat_volatilities[cat] = 0.0
            continue
        
        # Calculate month-to-month percentage changes
        pct_changes = []
        for i in range(1, len(recent_data)):
            prev = recent_data[i-1]
            curr = recent_data[i]
            
            if prev > 0:
                pct_change = abs((curr - prev) / prev) * 100
                pct_changes.append(pct_change)
        
        cat_volatilities[cat] = np.mean(pct_changes) if pct_changes else 0.0
    
    return cat_volatilities

def get_category_specific_weights(cat_volatilities):
    """
    Determine LSTM weight for EACH category based on its individual volatility
    
    GENIUS LOGIC:
    - Low volatility (<30%): Trust LSTM MORE (70% LSTM) - stable, learnable patterns
    - Medium volatility (30-60%): Balanced (50% LSTM) - moderate predictability  
    - High volatility (>60%): Trust Baseline MORE (30% LSTM) - too random, last month safer
    
    This is OPPOSITE of global weighting because:
    - Volatile categories (respiratory, skin_allergy) → baseline is safer than trying to predict chaos
    - Stable categories (pain_aches) → LSTM can actually learn the patterns
    
    Args:
        cat_volatilities: Dict of {category: volatility_percentage}
        
    Returns:
        dict: {category: {'lstm_weight': float, 'baseline_weight': float, 'volatility': float, 'mode': str}}
    """
    cat_weights = {}
    
    for cat, volatility in cat_volatilities.items():
        if volatility < 30:
            # Stable category - trust LSTM (patterns are learnable)
            lstm_weight = 0.7
            mode = "STABLE"
        elif volatility < 60:
            # Moderate volatility - balanced approach
            lstm_weight = 0.5
            mode = "MODERATE"
        else:
            # High volatility - trust baseline (last month is safer than predicting chaos)
            lstm_weight = 0.3
            mode = "VOLATILE"
        
        cat_weights[cat] = {
            'lstm_weight': lstm_weight,
            'baseline_weight': 1 - lstm_weight,
            'volatility': volatility,
            'mode': mode
        }
    
    return cat_weights

def calculate_volatility(df, categories, lookback_months=3):
    """
    Calculate recent volatility across all categories with recency weighting
    
    Args:
        df: DataFrame with historical data
        categories: List of category names
        lookback_months: Number of recent months to analyze
        
    Returns:
        float: Weighted average volatility percentage across categories
    """
    # Recency weights: most recent change gets more weight
    weights = [0.66, 0.34]  # For 3-month lookback: 66% recent, 34% older
    
    volatilities = []
    
    for cat in categories:
        recent_data = df[cat].tail(lookback_months + 1).values
        
        if len(recent_data) < 2:
            continue
            
        # Calculate month-to-month percentage changes
        pct_changes = []
        for i in range(1, len(recent_data)):
            prev = recent_data[i-1]
            curr = recent_data[i]
            
            if prev > 0:
                pct_change = abs((curr - prev) / prev) * 100
                pct_changes.append(pct_change)
        
        # Apply recency weights (most recent first)
        if pct_changes:
            pct_changes_rev = pct_changes[::-1]  # Reverse to get most recent first
            w = weights[:len(pct_changes_rev)]
            w_sum = sum(w)
            if w_sum > 0:
                weighted_vol = sum(pc * (wi/w_sum) for pc, wi in zip(pct_changes_rev, w))
                volatilities.append(weighted_vol)
    
    return np.mean(volatilities) if volatilities else 0.0

def calculate_momentum(df, categories, lookback_months=2):
    """
    Calculate momentum (rate of change) to detect trend direction
    
    Args:
        df: DataFrame with historical data
        categories: List of category names
        lookback_months: How far back to look for momentum (default: 2 months)
        
    Returns:
        float: Average momentum percentage across categories
    """
    if len(df) < lookback_months + 1:
        return 0.0
    
    momentums = []
    
    for cat in categories:
        if cat not in df.columns:
            continue
            
        # Get current month and month from lookback_months ago
        current = df[cat].iloc[-1]
        past = df[cat].iloc[-(lookback_months + 1)]
        
        if past > 0:
            momentum = ((current - past) / past) * 100
            momentums.append(momentum)
    
    return np.mean(momentums) if momentums else 0.0

def get_adaptive_weights(volatility):
    """
    Determine LSTM vs Baseline weights based on volatility
    
    STABLE (<20%): Trust baseline more (30% LSTM, 70% Baseline)
    MODERATE (20-50%): Balanced (50% LSTM, 50% Baseline)
    VOLATILE (>50%): Trust LSTM more (70% LSTM, 30% Baseline)
    
    Args:
        volatility: Average volatility percentage
        
    Returns:
        tuple: (lstm_weight, baseline_weight, mode_name)
    """
    if volatility < 20:
        return 0.3, 0.7, "STABLE"
    elif volatility < 50:
        return 0.5, 0.5, "MODERATE"
    else:
        return 0.7, 0.3, "VOLATILE"

def predict_next_month(historical_data_path=None, use_adaptive_weights=True):
    """
    Predict next month's cases using Hybrid Transfer Learning with Adaptive Weighting
    
    Args:
        historical_data_path: Path to CSV with historical monthly data (optional)
        use_adaptive_weights: Whether to use adaptive weighting based on volatility (default: True)
        
    Returns:
        dict: Predictions for each category and total
    """
    
    # Use default path if none provided
    if historical_data_path is None:
        historical_data_path = DATA_PATH
    
    print("=" * 80)
    print("PREDICTING NEXT MONTH")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading production model...")
    config, category_models = load_production_model()
    categories = config['categories']
    lookback = config['model_config']['lookback']
    default_blend_ratio = config['model_config']['blend_ratio']
    
    print(f"✓ Model: {config['model_name']} v{config['version']}")
    print(f"✓ Trained: {config['trained_date'][:10]}")
    
    # Load historical data
    print(f"\n📂 Loading historical data...")
    df = pd.read_csv(historical_data_path)
    df['month'] = pd.to_datetime(df['month'])
    
    last_month = df['month'].iloc[-1].strftime('%Y-%m')
    print(f"✓ Last month in data: {last_month}")
    print(f"✓ Total months: {len(df)}")
    
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} months of data, got {len(df)}")
    
    # CATEGORY-SPECIFIC ADAPTIVE WEIGHTING
    if use_adaptive_weights:
        print(f"\n📊 Analyzing category-specific volatility...")
        
        # Calculate volatility for EACH category (12-month lookback for stability)
        cat_volatilities = calculate_category_volatility(df, categories, lookback_months=12)
        cat_weights = get_category_specific_weights(cat_volatilities)
        
        print(f"\n{'Category':<30} {'Volatility':>12} {'Mode':>12} {'LSTM':>8} {'Baseline':>10}")
        print("-" * 78)
        
        for cat in categories:
            w = cat_weights[cat]
            print(f"{cat:<30} {w['volatility']:>11.1f}% {w['mode']:>12} {int(w['lstm_weight']*100):>7}% {int(w['baseline_weight']*100):>9}%")
        
        # Also calculate global volatility for summary
        global_volatility = calculate_volatility(df, categories, lookback_months=3)
        print(f"\n✓ Global volatility (3-month): {global_volatility:.1f}%")
        print(f"✓ Using category-specific weights (12-month analysis)")
        print(f"  → Stable categories (pain_aches): Trust LSTM MORE (70%)")
        print(f"  → Volatile categories (respiratory, skin_allergy): Trust Baseline MORE (70%)")
    else:
        # Fixed weights for all categories
        cat_weights = {}
        for cat in categories:
            cat_weights[cat] = {
                'lstm_weight': default_blend_ratio,
                'baseline_weight': 1 - default_blend_ratio,
                'volatility': None,
                'mode': "FIXED"
            }
        global_volatility = None
        print(f"\n✓ Using fixed weights: {int(default_blend_ratio*100)}% LSTM + {int(100-default_blend_ratio*100)}% Baseline")
    
    # STEP 1: Transfer learning predictions
    print(f"\n🤖 Running transfer learning predictions...")
    transfer_predictions = {}
    
    for cat in categories:
        # Get last lookback months for this category
        X_pred = df[cat].tail(lookback).values.reshape(1, lookback, 1)
        
        # Predict using fine-tuned model
        pred = float(category_models[cat].predict(X_pred, verbose=0)[0][0])
        pred = max(0, pred)  # Ensure non-negative
        
        transfer_predictions[cat] = pred
    
    transfer_total = sum(transfer_predictions.values())
    print(f"  Transfer learning total: {transfer_total:.1f} cases")
    
    # STEP 2: Baseline predictions (last month)
    print(f"\n📊 Getting baseline (last month)...")
    baseline_predictions = {}
    
    for cat in categories:
        baseline_predictions[cat] = float(df[cat].iloc[-1])
    
    baseline_total = sum(baseline_predictions.values())
    print(f"  Baseline total: {baseline_total:.1f} cases")
    
    # STEP 3: Hybrid blend with CATEGORY-SPECIFIC adaptive weights
    print(f"\n🔀 Blending predictions (category-specific weights)...")
    hybrid_predictions = {}
    
    for cat in categories:
        cat_lstm_weight = cat_weights[cat]['lstm_weight']
        cat_baseline_weight = cat_weights[cat]['baseline_weight']
        
        hybrid_predictions[cat] = (
            cat_lstm_weight * transfer_predictions[cat] + 
            cat_baseline_weight * baseline_predictions[cat]
        )
    
    hybrid_total = sum(hybrid_predictions.values())
    
    # Round to nearest integer
    hybrid_predictions_rounded = {k: round(v) for k, v in hybrid_predictions.items()}
    
    print(f"  Hybrid total: {hybrid_total:.1f} cases")
    
    # Calculate average weights for summary
    avg_lstm_weight = np.mean([w['lstm_weight'] for w in cat_weights.values()])
    avg_baseline_weight = 1 - avg_lstm_weight
    
    # STEP 4: Prepare results
    results = {
        'prediction_date': pd.Timestamp.now().isoformat(),
        'predicting_for': (df['month'].iloc[-1] + pd.DateOffset(months=1)).strftime('%Y-%m-%d'),
        'last_data_month': last_month,
        'model_version': config['version'],
        'adaptive_weighting': {
            'enabled': use_adaptive_weights,
            'method': 'category_specific' if use_adaptive_weights else 'fixed',
            'global_volatility': round(global_volatility, 2) if global_volatility is not None else None,
            'avg_lstm_weight': round(avg_lstm_weight, 2),
            'avg_baseline_weight': round(avg_baseline_weight, 2),
            'category_weights': {cat: {
                'volatility': round(w['volatility'], 1) if w['volatility'] is not None else None,
                'mode': w['mode'],
                'lstm_weight': round(w['lstm_weight'], 2),
                'baseline_weight': round(w['baseline_weight'], 2)
            } for cat, w in cat_weights.items()}
        },
        'predictions': {
            'by_category': hybrid_predictions_rounded,
            'total': round(hybrid_total)
        },
        'components': {
            'transfer_learning': {k: round(v, 1) for k, v in transfer_predictions.items()},
            'baseline': baseline_predictions,
            'transfer_total': round(transfer_total, 1),
            'baseline_total': round(baseline_total, 1)
        }
    }
    
    # Display results
    print(f"\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    print(f"\nPredicting for: {results['predicting_for']}")
    
    if use_adaptive_weights and global_volatility is not None:
        print(f"\n🎯 Category-Specific Adaptive Weighting:")
        print(f"   Global volatility: {global_volatility:.1f}%")
        print(f"   Average weights: {int(avg_lstm_weight*100)}% LSTM + {int(avg_baseline_weight*100)}% Baseline")
        print(f"   (Individual category weights vary based on their volatility)")
    
    print(f"\n{'Category':<30} {'Transfer':<12} {'Baseline':<12} {'Hybrid':<12} {'LSTM%':<8}")
    print("-" * 88)
    
    for cat in categories:
        cat_lstm_pct = int(cat_weights[cat]['lstm_weight'] * 100)
        print(f"{cat:<30} {transfer_predictions[cat]:<12.1f} {baseline_predictions[cat]:<12.1f} {hybrid_predictions_rounded[cat]:<12} {cat_lstm_pct:<8}%")
    
    print("-" * 88)
    print(f"{'TOTAL':<30} {transfer_total:<12.1f} {baseline_total:<12.1f} {round(hybrid_total):<12}")
    
    print(f"\n💾 Saving predictions...")
    with open('predictions/latest_prediction.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Saved: predictions/latest_prediction.json")
    
    return results

if __name__ == '__main__':
    # Run prediction
    results = predict_next_month()
    print(f"\n✅ Prediction complete!")
