import pandas as pd
import numpy as np
import sys
from statsmodels.tsa.arima.model import ARIMA
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
INPUT_FILE = "./data/PoC_Outliers_Treated.csv"
OUTPUT_FILE = "./data/PoC_Forecast_Results.csv"
SUMMARY_FILE = "./data/PoC_Forecast_Summary.json"

# Forecast 12 months (2026-01 to 2026-12)
FORECAST_HORIZON = 12

# Split historical data for WMAPE calculation
# Using 20 months for training and the last 4 for testing (2025-09 to 2025-12)
TRAIN_SIZE = 20

# Model Orders (Fixed for PoC comparison)
# ARIMA: Non-seasonal (p, d, q)
ARIMA_ORDER = (1, 1, 1)
# SARIMA: Seasonal (p, d, q) x (P, D, Q, s) - s=12 for monthly seasonality
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 12)


def fail(msg):
    """Prints an error message and exits."""
    print(f"[ERROR] {msg}")
    sys.exit(1)


def calculate_wmape(actual, predicted):
    """
    Calculates Weighted Mean Absolute Percentage Error (WMAPE).
    Formula: SUM(|Actual - Predicted|) / SUM(|Actual|)
    """
    if actual.sum() == 0:
        return 1.0  # Cannot divide by zero; return 100% error
    
    numerator = np.abs(actual - predicted).sum()
    denominator = np.abs(actual).sum()
    
    return (numerator / denominator) * 100


def run_forecast_model(train_data, test_data, model_type, order, seasonal_order=None):
    """Trains and tests a model, and produces a 12-month forecast."""

    if model_type == 'ARIMA':
        model = ARIMA(train_data, order=order)
    elif model_type == 'SARIMA':
        model = ARIMA(train_data, order=order, seasonal_order=seasonal_order)
    else:
        raise ValueError("Invalid model type")

    # Train model
    try:
        model_fit = model.fit()
    except Exception as e:
        # Catch exceptions during fitting (like non-convergence)
        print(f"[WARN] {model_type} failed to fit: {e}")
        return {
            'wmape': 1000.0,  # Assign high error on failure
            'forecast_df': pd.DataFrame(),
            'model_fit': None
        }

    # 1. Calculate WMAPE on test set
    start_test = test_data.index[0]
    end_test = test_data.index[-1]
    
    # Predict on test data
    pred_test = model_fit.get_prediction(start=start_test, end=end_test)
    
    # Check if prediction was successful
    if pred_test is None or pred_test.predicted_mean.empty:
        wmape = 1000.0
    else:
        # NOTE: pred_test.predicted_mean and test_data have PeriodIndex, which aligns automatically
        wmape = calculate_wmape(test_data, pred_test.predicted_mean)
        
    # 2. Generate future forecast (FORECAST_HORIZON months)
    
    # Determine forecast period dates for MonthKey creation
    # FIX APPLIED IN PREVIOUS STEP: Convert PeriodIndex object to timestamp before date arithmetic
    last_date_of_train = train_data.index[-1].to_timestamp()
    
    # Calculate the first date of the forecast horizon
    start_forecast = (last_date_of_train + relativedelta(months=1))

    # Create month keys for the forecast horizon
    forecast_dates = [start_forecast + relativedelta(months=i) for i in range(FORECAST_HORIZON)]
    forecast_month_keys = [d.strftime('%Y-%m') for d in forecast_dates]
    
    # Generate the forecast
    forecast_results = model_fit.get_forecast(steps=FORECAST_HORIZON)
    
    # Extract mean, lower and upper confidence intervals (95%)
    forecast_mean = forecast_results.predicted_mean.values
    confidence_intervals = forecast_results.conf_int(alpha=0.05)
    
    forecast_df = pd.DataFrame({
        'MonthKey': forecast_month_keys,
        'ForecastMean': forecast_mean,
        'ForecastLowerCI': confidence_intervals.iloc[:, 0].values,
        'ForecastUpperCI': confidence_intervals.iloc[:, 1].values,
    })
    
    return {
        'wmape': wmape,
        'forecast_df': forecast_df,
        'model_fit': model_fit
    }


def main():

    # ----------------------------------------------------------------
    # 1. LOAD DATA AND PREPARE SERIES
    # ----------------------------------------------------------------
    if not pd.io.common.file_exists(INPUT_FILE):
        fail(f"Input file not found: {INPUT_FILE}. Run 04_outliers_GPT.py first.")

    df = pd.read_csv(INPUT_FILE)
    df['MonthKey'] = pd.to_datetime(df['MonthKey'])
    
    # Use the outlier-cleaned series for forecasting
    ts = df.set_index('MonthKey')['NetFlow_Clean']
    ts.index = ts.index.to_period('M') # Convert to Monthly PeriodIndex for statsmodels

    if len(ts) < TRAIN_SIZE + 4:
        fail(f"Insufficient data ({len(ts)} months). Need at least {TRAIN_SIZE + 4} months for training/testing split.")

    # Split into train (20 months) and test (4 months)
    train = ts[:TRAIN_SIZE]
    test = ts[TRAIN_SIZE:]
    
    print(f"[INFO] Data loaded. Training on {len(train)} months, Testing on {len(test)} months.")

    # ----------------------------------------------------------------
    # 2. RUN ARIMA MODEL
    # ----------------------------------------------------------------
    print(f"\n[INFO] Running ARIMA {ARIMA_ORDER}...")
    arima_results = run_forecast_model(
        train_data=train,
        test_data=test,
        model_type='ARIMA',
        order=ARIMA_ORDER
    )

    # ----------------------------------------------------------------
    # 3. RUN SARIMA MODEL
    # ----------------------------------------------------------------
    print(f"[INFO] Running SARIMA {SARIMA_ORDER} x {SARIMA_SEASONAL_ORDER}...")
    sarima_results = run_forecast_model(
        train_data=train,
        test_data=test,
        model_type='SARIMA',
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER
    )

    # ----------------------------------------------------------------
    # 4. COMPARE RESULTS AND SELECT WINNER (TC-05)
    # ----------------------------------------------------------------
    wmape_arima = arima_results['wmape']
    wmape_sarima = sarima_results['wmape']
    
    if wmape_arima < wmape_sarima:
        winner_model = "ARIMA"
        final_forecast_df = arima_results['forecast_df']
    else:
        winner_model = "SARIMA"
        final_forecast_df = sarima_results['forecast_df']
        
    if final_forecast_df.empty:
        print("[ERROR] Both models failed to produce a forecast. Check model fitting warnings.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 5. ASSEMBLE FINAL OUTPUT
    # ----------------------------------------------------------------
    
    # Join historical data (original and cleaned net flow) with the forecast
    final_output_df = df[['MonthKey', 'NetFlowExternal', 'NetFlow_Clean', 'TreatmentTag', 'IsOutlier']].copy()
    
    # Format historical MonthKey as YYYY-MM string
    final_output_df['MonthKey'] = final_output_df['MonthKey'].dt.strftime('%Y-%m')
    
    # Merge and fill missing forecast columns in history with NaN
    final_output_df = pd.merge(
        final_output_df, 
        final_forecast_df, 
        on='MonthKey', 
        how='outer'
    )
    
    # Final cleanup for display
    final_output_df = final_output_df.sort_values(by='MonthKey').reset_index(drop=True)
    
    # Ensure forecast values are rounded
    for col in ['ForecastMean', 'ForecastLowerCI', 'ForecastUpperCI']:
        final_output_df[col] = final_output_df[col].round(2)

    # Save final results CSV
    final_output_df.to_csv(OUTPUT_FILE, index=False)
    
    # Prepare JSON Summary for LLM Explainability (TC-06)
    
    # FIX APPLIED HERE: Ensure MonthKey in outliers_detected is string for JSON serialization
    outliers_for_json = df[df['IsOutlier']][['MonthKey', 'NetFlowExternal', 'TreatmentTag']].copy()
    outliers_for_json['MonthKey'] = outliers_for_json['MonthKey'].dt.strftime('%Y-%m')

    # Get winner parameters
    winner_fit = arima_results['model_fit'] if winner_model == 'ARIMA' else sarima_results['model_fit']
    winner_params = winner_fit.params.to_dict() if winner_fit else {}

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "wmape_arima": round(wmape_arima, 3),
            "wmape_sarima": round(wmape_sarima, 3),
            "wmape_threshold": "< 20.0",
            "winner_model": winner_model,
            "winner_params": winner_params
        },
        "forecast_period": {
            "start": final_forecast_df['MonthKey'].min(),
            "end": final_forecast_df['MonthKey'].max(),
            "horizon_months": FORECAST_HORIZON
        },
        "forecast_results": final_forecast_df.to_dict('records'),
        "outliers_detected": outliers_for_json.to_dict('records')
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary_data, f, indent=4)


    # ----------------------------------------------------------------
    # 6. LOG SUMMARY (Final Check)
    # ----------------------------------------------------------------
    print("\n=======================================================")
    print("✅ 05_forecast_GPT.py: Forecasting Completed")
    print("=======================================================")
    print(f"Training Period: {train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')}")
    print(f"Testing Period: {test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')}")
    print(f"Forecast Horizon: {FORECAST_HORIZON} months ({summary_data['forecast_period']['start']} - {summary_data['forecast_period']['end']})")
    print("-------------------------------------------------------")
    print("WMAPE Results:")
    print(f"  ARIMA WMAPE: {wmape_arima:.3f}%")
    print(f"  SARIMA WMAPE: {wmape_sarima:.3f}%")
    print(f"🏆 Winner Model: {winner_model}")
    print("-------------------------------------------------------")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"JSON Summary saved to: {SUMMARY_FILE} (For LLM Explainability)")
    print("=======================================================")


if __name__ == "__main__":
    main()