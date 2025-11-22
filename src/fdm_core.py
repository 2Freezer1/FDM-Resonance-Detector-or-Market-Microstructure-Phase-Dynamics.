"""
Fragility Detection Model (FDM) v9.4 - Resonant Inertia Implementation

Theoretical Basis:
E_Det = k * (Delta_rho)^2 * (Delta_S / N_eff) + delta_Q

Implements adaptive SARIMAX grid search to minimize Information Criterion (AIC)
as a proxy for thermodynamic energy cost in complex systems.
"""
import pandas as pd
import numpy as np
import time
import requests
import warnings
import itertools
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration Constants ---
TICKER = "BTCUSDT"
INTERVAL = "1h"
POLLING_INTERVAL_SEC = 1800 
WARMUP_PERIOD = 50
FORECAST_HORIZON = 10
DATA_LIMIT = 100
VOLATILITY_WINDOW = 5
AIC_THRESHOLD_BUFFER = 10.0  # NEW: Only re-search if AIC gets worse by this amount

# --- SARIMAX CONFIG ---
P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)
SARIMAX_SEASONAL = (0, 0, 0, 0)

warnings.filterwarnings("ignore")

class FDM_Trader:
    def __init__(self):
        self.risk = 5.0
        self.last_signal = None
        self.signal_streak = 0
        self.current_order = (1, 1, 1) # Default starting order
        self.last_aic = np.inf
        self.session = self._init_session()

    def _init_session(self):
        """Creates a robust HTTP session with retries."""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def log_event(self, msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    def log_trade(self, signal, price, trend):
        msg = f"TRADE: {signal} | Price: {price:,.2f} | Risk: {self.risk:.2f}/10 | Trend: {trend} | Streak: {self.signal_streak}h"
        print("\n" + "="*80)
        print(f"!!! RESONANCE SIGNAL DETECTED !!! - {msg}")
        print("="*80 + "\n")

    def fetch_metrics(self):
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': TICKER, 'interval': INTERVAL, 'limit': DATA_LIMIT + WARMUP_PERIOD}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data: return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tbba', 'tbqa', 'ignore'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.set_index('time').sort_index()
            df = df.astype({'close': float, 'volume': float})
            return df[['close', 'volume']]
            
        except Exception as e:
            self.log_event(f"Data Fetch Error: {e}")
            return pd.DataFrame()

    def fit_sarimax(self, series, exog, order):
        """Helper to fit a single model and return it along with AIC."""
        try:
            model = SARIMAX(
                series, exog=exog, order=order, 
                seasonal_order=SARIMAX_SEASONAL,
                enforce_stationarity=False, enforce_invertibility=False
            )
            model_fit = model.fit(disp=False, method='lbfgs', maxiter=50) # Optimized maxiter
            return model_fit, model_fit.aic
        except:
            return None, np.inf

    def adaptive_model_selection(self, series, exog_data):
        """
        OPTIMIZATION: Inertia Check.
        First, try the existing order. If it's stable (AIC hasn't spiked), keep it.
        If it degrades, perform the expensive Grid Search.
        """
        # 1. Test Current Model (Inertia)
        current_model, current_new_aic = self.fit_sarimax(series, exog_data, self.current_order)
        
        # If the model is still good (AIC hasn't exploded), stick with it to save compute
        if current_new_aic < self.last_aic + AIC_THRESHOLD_BUFFER:
            self.last_aic = current_new_aic
            return self.current_order, current_model

        # 2. If degraded, perform Grid Search (Phase Transition)
        self.log_event(f"Resonance Shift Detected (AIC Spike). Recalibrating Matrix...")
        best_aic = np.inf
        best_order = self.current_order
        best_model = current_model

        pdq_combinations = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
        
        for order in pdq_combinations:
            # Skip the one we just tested
            if order == self.current_order: continue 
            
            model, aic = self.fit_sarimax(series, exog_data, order)
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_model = model
        
        self.last_aic = best_aic
        self.current_order = best_order
        self.log_event(f"New Stable State Found: {best_order} (AIC: {best_aic:.2f})")
        return best_order, best_model

    def run(self):
        self.log_event(f"--- FDM v9.4 (Resonant Inertia) Starting ---")
        
        while True:
            try:
                metrics = self.fetch_metrics()
                if metrics.empty or len(metrics) < DATA_LIMIT + VOLATILITY_WINDOW:
                    time.sleep(60)
                    continue

                # --- Feature Engineering ---
                metrics['volatility'] = metrics['close'].pct_change().rolling(VOLATILITY_WINDOW).std()
                metrics = metrics.dropna()
                
                # Normalization (Mean Scaling)
                vol_mean = metrics['volatility'].mean()
                metrics['volatility'] = metrics['volatility'] / (vol_mean if vol_mean > 0 else 1)
                
                data = metrics.iloc[-DATA_LIMIT:]
                series = data['close']
                exog_data = data[['volume', 'volatility']]

                # --- Adaptive Modeling ---
                optimal_order, model_fit = self.adaptive_model_selection(series, exog_data)

                # --- Extended Forecast with Smoothing ---
                # OPTIMIZATION: Use rolling mean for future exog inputs (Smoother Consensus)
                recent_exog = exog_data.rolling(window=3).mean().iloc[-1]
                future_exog = pd.DataFrame([recent_exog] * FORECAST_HORIZON, columns=exog_data.columns)
                future_exog.index = range(len(series), len(series) + FORECAST_HORIZON) # Fix index alignment

                forecast = model_fit.get_forecast(steps=FORECAST_HORIZON, exog=future_exog)
                predicted_price = forecast.predicted_mean.iloc[-1]
                current_price = series.iloc[-1]
                
                delta = predicted_price - current_price
                stdev = series.diff().std()

                # --- Signal Logic ---
                base_signal = "HOLD"
                trend = "STABLE"

                if abs(delta) > stdev * 0.5:
                    if delta > 0:
                        trend = "UPWARD"
                        base_signal = "BUY"
                        self.risk = min(10.0, self.risk * 1.2)
                    else:
                        trend = "DOWNWARD"
                        base_signal = "SELL"
                        self.risk *= 0.8
                else:
                    self.risk = max(1.0, self.risk * 0.95)

                # --- Logging ---
                self.log_event(f"Model: {optimal_order} | Price: ${current_price:,.2f} | FC (T+10): {delta:+.2f} | AIC: {self.last_aic:.1f} | Risk: {self.risk:.1f}")

                # --- Trade Execution ---
                trade_signal = "HOLD"
                if self.risk >= 8.5:
                    if base_signal == self.last_signal:
                        self.signal_streak += 1
                    else:
                        self.signal_streak = 1
                        self.last_signal = base_signal
                    
                    if self.signal_streak >= 3 and base_signal != "HOLD":
                        trade_signal = base_signal
                else:
                    self.signal_streak = 0
                    self.last_signal = None

                if trade_signal != "HOLD":
                    self.log_trade(trade_signal, current_price, trend)
                    self.signal_streak = 0 # Reset after trade
                    self.risk = 5.0

                time.sleep(POLLING_INTERVAL_SEC)

            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                self.log_event(f"Critical Loop Error: {e}")
                time.sleep(60)

if __name__ == '__main__':
    bot = FDM_Trader()
    bot.run()
