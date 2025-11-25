"""
Real-Time Prediction Adjustment System
======================================
Start with model prediction, adjust as the month progresses based on actual trend

GENIUS APPROACH:
- Day 1: Use model prediction (LSTM + Baseline blend)
- Week 1: Check if trend is up/down, adjust prediction
- Week 2: Refine adjustment based on 2-week trend
- Week 3: Further refine
- Week 4: Final adjustment

This catches turning points EARLY instead of waiting until month-end!
"""

import json
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from predict import (
    load_production_model, 
    calculate_category_volatility, 
    get_category_specific_weights
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'monthly_data_8cat_no_covid.csv')

def adjust_prediction_realtime(initial_prediction, current_partial_cases, last_month_total, 
                                days_elapsed, days_in_month=30, category_weights=None):
    """
    Adjust prediction based on actual cases so far this month
    
    LOGIC:
    - If current pace is 20%+ higher than last month → Spike detected, adjust UP
    - If current pace is 20%+ lower than last month → Drop detected, adjust DOWN
    - Otherwise → Keep original prediction with minor adjustment
    
    Args:
        initial_prediction: Model's initial prediction for the month
        current_partial_cases: Actual cases counted so far this month
        last_month_total: Total cases from last month (baseline)
        days_elapsed: How many days have passed this month
        days_in_month: Total days in the month (default 30)
        category_weights: Optional dict of category-specific volatility info
        
    Returns:
        dict: Adjusted prediction with confidence and reasoning
    """
    
    # Calculate expected pace based on last month
    expected_pace = (last_month_total / days_in_month) * days_elapsed
    
    # Calculate actual pace percentage vs expected
    if expected_pace > 0:
        pace_ratio = current_partial_cases / expected_pace
    else:
        pace_ratio = 1.0
    
    # Project to end of month based on current pace
    projected_total = (current_partial_cases / days_elapsed) * days_in_month
    
    # Determine confidence based on how much of month has passed
    confidence = min(days_elapsed / days_in_month * 100, 95)  # Max 95% confidence
    
    # ADJUSTMENT LOGIC
    adjustment_reason = ""
    adjusted_prediction = initial_prediction
    
    if pace_ratio > 1.3:
        # MAJOR SPIKE DETECTED (>30% higher than expected)
        # Trust the actual trend MORE than the model
        blend_weight = min(days_elapsed / 7, 0.8)  # Up to 80% trust in actual pace after week 1
        adjusted_prediction = blend_weight * projected_total + (1 - blend_weight) * initial_prediction
        adjustment_reason = f"SPIKE DETECTED: Cases running {(pace_ratio-1)*100:.0f}% higher than expected. Adjusting UP."
        trend = "SPIKE"
        
    elif pace_ratio > 1.15:
        # Moderate spike (15-30% higher)
        blend_weight = min(days_elapsed / 14, 0.5)  # Up to 50% trust in actual pace after week 2
        adjusted_prediction = blend_weight * projected_total + (1 - blend_weight) * initial_prediction
        adjustment_reason = f"Moderate increase: Cases running {(pace_ratio-1)*100:.0f}% higher than expected. Minor adjustment UP."
        trend = "RISING"
        
    elif pace_ratio < 0.7:
        # MAJOR DROP DETECTED (>30% lower than expected)
        blend_weight = min(days_elapsed / 7, 0.8)  # Up to 80% trust in actual pace after week 1
        adjusted_prediction = blend_weight * projected_total + (1 - blend_weight) * initial_prediction
        adjustment_reason = f"DROP DETECTED: Cases running {(1-pace_ratio)*100:.0f}% lower than expected. Adjusting DOWN."
        trend = "DROP"
        
    elif pace_ratio < 0.85:
        # Moderate drop (15-30% lower)
        blend_weight = min(days_elapsed / 14, 0.5)  # Up to 50% trust in actual pace after week 2
        adjusted_prediction = blend_weight * projected_total + (1 - blend_weight) * initial_prediction
        adjustment_reason = f"Moderate decrease: Cases running {(1-pace_ratio)*100:.0f}% lower than expected. Minor adjustment DOWN."
        trend = "FALLING"
        
    else:
        # On track (within 15% of expected)
        # Minor adjustment towards actual pace
        blend_weight = min(days_elapsed / 21, 0.3)  # Up to 30% trust in actual pace after week 3
        adjusted_prediction = blend_weight * projected_total + (1 - blend_weight) * initial_prediction
        adjustment_reason = f"On track: Cases within expected range. Keeping model prediction with minor adjustment."
        trend = "STABLE"
    
    return {
        'initial_prediction': round(initial_prediction, 1),
        'adjusted_prediction': round(adjusted_prediction),
        'current_partial_cases': current_partial_cases,
        'days_elapsed': days_elapsed,
        'days_in_month': days_in_month,
        'progress_pct': round(days_elapsed / days_in_month * 100, 1),
        'expected_pace': round(expected_pace, 1),
        'pace_ratio': round(pace_ratio, 2),
        'projected_total': round(projected_total, 1),
        'trend': trend,
        'adjustment_amount': round(adjusted_prediction - initial_prediction, 1),
        'adjustment_pct': round((adjusted_prediction - initial_prediction) / initial_prediction * 100, 1) if initial_prediction > 0 else 0,
        'confidence': round(confidence, 1),
        'reason': adjustment_reason
    }

def predict_with_realtime_adjustment(current_partial_data=None, days_elapsed=None):
    """
    Make prediction and optionally adjust based on current month's partial data
    
    Args:
        current_partial_data: Dict of {category: partial_cases_so_far} (optional)
        days_elapsed: How many days have passed this month (optional)
        
    Returns:
        dict: Prediction with optional real-time adjustment
    """
    
    print("=" * 80)
    print("REAL-TIME PREDICTION WITH ADJUSTMENT")
    print("=" * 80)
    
    # Load model and make initial prediction
    print("\n📦 Loading production model...")
    config, category_models = load_production_model()
    categories = config['categories']
    lookback = config['model_config']['lookback']
    
    # Load historical data
    df = pd.read_csv(DATA_PATH)
    df['month'] = pd.to_datetime(df['month'])
    
    last_month = df['month'].iloc[-1].strftime('%Y-%m')
    print(f"✓ Last complete month: {last_month}")
    
    # Calculate category-specific weights
    cat_volatilities = calculate_category_volatility(df, categories, lookback_months=12)
    cat_weights = get_category_specific_weights(cat_volatilities)
    
    # Make initial predictions
    print(f"\n🤖 Making initial predictions...")
    initial_predictions = {}
    last_month_totals = {}
    
    for cat in categories:
        # LSTM prediction
        X_pred = df[cat].tail(lookback).values.reshape(1, lookback, 1)
        lstm_pred = float(category_models[cat].predict(X_pred, verbose=0)[0][0])
        lstm_pred = max(0, lstm_pred)
        
        # Baseline
        baseline_pred = float(df[cat].iloc[-1])
        last_month_totals[cat] = baseline_pred
        
        # Hybrid with category-specific weights
        cat_lstm_weight = cat_weights[cat]['lstm_weight']
        cat_baseline_weight = cat_weights[cat]['baseline_weight']
        
        initial_predictions[cat] = (cat_lstm_weight * lstm_pred) + (cat_baseline_weight * baseline_pred)
    
    initial_total = sum(initial_predictions.values())
    last_month_total = sum(last_month_totals.values())
    
    print(f"✓ Initial prediction: {initial_total:.0f} cases")
    print(f"✓ Last month total: {last_month_total:.0f} cases")
    
    # Real-time adjustment if partial data provided
    if current_partial_data is not None and days_elapsed is not None:
        print(f"\n📊 REAL-TIME ADJUSTMENT (Day {days_elapsed})...")
        
        adjustments = {}
        adjusted_predictions = {}
        
        for cat in categories:
            if cat in current_partial_data:
                adjustment = adjust_prediction_realtime(
                    initial_prediction=initial_predictions[cat],
                    current_partial_cases=current_partial_data[cat],
                    last_month_total=last_month_totals[cat],
                    days_elapsed=days_elapsed,
                    days_in_month=30
                )
                adjustments[cat] = adjustment
                adjusted_predictions[cat] = adjustment['adjusted_prediction']
            else:
                adjusted_predictions[cat] = round(initial_predictions[cat])
        
        adjusted_total = sum(adjusted_predictions.values())
        
        # Overall summary
        total_adjustment = adjust_prediction_realtime(
            initial_prediction=initial_total,
            current_partial_cases=sum(current_partial_data.values()),
            last_month_total=last_month_total,
            days_elapsed=days_elapsed,
            days_in_month=30
        )
        
        print(f"\n{'=' * 80}")
        print("ADJUSTMENT SUMMARY")
        print(f"{'=' * 80}")
        print(f"\n📅 Progress: Day {days_elapsed}/30 ({total_adjustment['progress_pct']:.1f}%)")
        print(f"📊 Trend: {total_adjustment['trend']}")
        print(f"📈 Pace: {total_adjustment['pace_ratio']:.2f}x expected")
        print(f"🎯 Confidence: {total_adjustment['confidence']:.1f}%")
        print(f"\n{total_adjustment['reason']}")
        
        print(f"\n{'Category':<30} {'Initial':>10} {'Current':>10} {'Adjusted':>10} {'Change':>10}")
        print("-" * 80)
        
        for cat in categories:
            if cat in adjustments:
                adj = adjustments[cat]
                change = adj['adjustment_amount']
                print(f"{cat:<30} {adj['initial_prediction']:>10.0f} {adj['current_partial_cases']:>10.0f} {adj['adjusted_prediction']:>10.0f} {change:>+10.0f}")
            else:
                print(f"{cat:<30} {initial_predictions[cat]:>10.0f} {'N/A':>10} {round(initial_predictions[cat]):>10.0f} {'N/A':>10}")
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {initial_total:>10.0f} {sum(current_partial_data.values()):>10.0f} {adjusted_total:>10.0f} {adjusted_total - initial_total:>+10.0f}")
        
        return {
            'prediction_type': 'real_time_adjusted',
            'days_elapsed': days_elapsed,
            'days_in_month': 30,
            'progress_pct': total_adjustment['progress_pct'],
            'confidence': total_adjustment['confidence'],
            'trend': total_adjustment['trend'],
            'initial_prediction': round(initial_total),
            'adjusted_prediction': adjusted_total,
            'adjustment_amount': round(adjusted_total - initial_total),
            'adjustment_pct': round((adjusted_total - initial_total) / initial_total * 100, 1),
            'by_category': {
                'initial': {k: round(v) for k, v in initial_predictions.items()},
                'adjusted': adjusted_predictions,
                'current_partial': current_partial_data
            },
            'adjustments': adjustments
        }
    
    else:
        # No partial data, return initial prediction only
        print(f"\n✓ No partial data provided - returning initial prediction")
        
        return {
            'prediction_type': 'initial',
            'confidence': 50.0,  # Low confidence at start of month
            'initial_prediction': round(initial_total),
            'by_category': {k: round(v) for k, v in initial_predictions.items()}
        }

if __name__ == '__main__':
    # Example usage
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Initial prediction (no adjustment)")
    print("=" * 80)
    result1 = predict_with_realtime_adjustment()
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Real-time adjustment after Week 1 (7 days)")
    print("=" * 80)
    # Simulate Week 1 data (example: cases are spiking)
    week1_data = {
        'respiratory': 20,  # Higher than expected
        'gastrointestinal': 5,
        'pain_aches': 18,
        'skin_allergy': 4,
        'injury_trauma': 2,
        'neurological_psychological': 3,
        'cardiovascular': 1,
        'fever_general': 5
    }
    result2 = predict_with_realtime_adjustment(current_partial_data=week1_data, days_elapsed=7)
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Real-time adjustment after Week 2 (14 days)")
    print("=" * 80)
    # Simulate Week 2 data (spike continues)
    week2_data = {
        'respiratory': 38,  # Still high
        'gastrointestinal': 9,
        'pain_aches': 32,
        'skin_allergy': 8,
        'injury_trauma': 4,
        'neurological_psychological': 5,
        'cardiovascular': 1,
        'fever_general': 8
    }
    result3 = predict_with_realtime_adjustment(current_partial_data=week2_data, days_elapsed=14)
    
    print(f"\n✅ Real-time adjustment system ready!")
