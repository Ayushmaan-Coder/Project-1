import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from functools import wraps
import time
from pathlib import Path
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from enum import Enum
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'TIME_RANGES': {
        "1D": {"interval": "1m", "period": "1d", "update_interval": 60},
        "5D": {"interval": "5m", "period": "5d", "update_interval": 300},
        "1M": {"interval": "30m", "period": "1mo", "update_interval": 1800},
        "3M": {"interval": "1h", "period": "3mo", "update_interval": 3600},
        "6M": {"interval": "1d", "period": "6mo", "update_interval": 86400}
    },
    'LSTM_MODELS': {
        'vanilla': {
            'name': "Vanilla LSTM",
            'lookback': 60,
            'batch_size': 32,
            'epochs': 20
        },
        'stacked': {
            'name': "Stacked LSTM",
            'lookback': 60,
            'batch_size': 32,
            'epochs': 20
        },
        'bidirectional': {
            'name': "Bidirectional LSTM",
            'lookback': 60,
            'batch_size': 32,
            'epochs': 20
        },
        'seq2seq': {
            'name': "Sequence-to-Sequence",
            'lookback': 60,
            'batch_size': 32,
            'epochs': 20
        },
        'hybrid': {
            'name': "Hybrid LSTM",
            'lookback': 60,
            'batch_size': 32,
            'epochs': 20
        }
    },
    'DEFAULT_TIMEZONE': 'Asia/Kolkata',
    'CHART_HEIGHT': 500,
    'DEFAULT_REFRESH': True,
    'TABLE_PAGE_SIZE': 50,
    'MAX_RECORDS': 1000,
    'CACHE_TTL': 300,
    'API_RATE_LIMIT': 5,
    'SYMBOLS': [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
        'BHARTIARTL.NS', 'SBIN.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS',
        'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS',
        'TITAN.NS', 'BAJAJFINSV.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'TECHM.NS', 'POWERGRID.NS',
        'NTPC.NS', 'M&M.NS', 'ONGC.NS', 'HDFCLIFE.NS', 'TATASTEEL.NS'
    ]
}

# Technical Indicators Functions
def calculate_moving_average(df, window=20):
    """Calculate moving average for the given window."""
    df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=window).std()
    df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=window).std()
    return df

def calculate_macd(df):
    """Calculate MACD indicator."""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def calculate_all_indicators(df):
    """Calculate all technical indicators at once."""
    df = df.copy()
    
    # Moving Averages
    for window in [20, 50, 200]:
        df = calculate_moving_average(df, window)
    
    # RSI
    df = calculate_rsi(df)
    
    # Bollinger Bands
    df = calculate_bollinger_bands(df)
    
    # MACD
    df = calculate_macd(df)
    
    return df

# LSTM Model Class
class ModelType(Enum):
    VANILLA = "vanilla"
    STACKED = "stacked"
    BIDIRECTIONAL = "bidirectional"
    SEQ2SEQ = "seq2seq"
    HYBRID = "hybrid"

class LSTMModel:
    def __init__(self, model_type: ModelType, lookback: int = 60):
        self.model_type = model_type
        self.lookback = lookback
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == ModelType.VANILLA:
            return self._build_vanilla_lstm()
        elif self.model_type == ModelType.STACKED:
            return self._build_stacked_lstm()
        elif self.model_type == ModelType.BIDIRECTIONAL:
            return self._build_bidirectional_lstm()
        elif self.model_type == ModelType.SEQ2SEQ:
            return self._build_seq2seq_lstm()
        elif self.model_type == ModelType.HYBRID:
            return self._build_hybrid_lstm()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def _build_vanilla_lstm(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _build_stacked_lstm(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _build_bidirectional_lstm(self):
        model = Sequential([
            Bidirectional(LSTM(units=50, return_sequences=True), 
                         input_shape=(self.lookback, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(units=50)),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _build_seq2seq_lstm(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(1))
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _build_hybrid_lstm(self):
        price_input = Input(shape=(self.lookback, 1))
        x1 = LSTM(50, return_sequences=True)(price_input)
        x1 = Dropout(0.2)(x1)
        x1 = LSTM(50)(x1)
        price_output = Dropout(0.2)(x1)

        tech_input = Input(shape=(self.lookback, 5))
        x2 = LSTM(25)(tech_input)
        tech_output = Dropout(0.2)(x2)

        combined = Concatenate()([price_output, tech_output])
        output = Dense(1)(combined)

        model = Model(inputs=[price_input, tech_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def prepare_data(self, price_data: np.ndarray, tech_data=None):
        if self.model_type == ModelType.HYBRID and tech_data is not None:
            return self._prepare_hybrid_data(price_data, tech_data)
        return self._prepare_standard_data(price_data)

    def _prepare_standard_data(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, X, y, validation_split=0.2, epochs=20, batch_size=32):
        return self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def predict(self, X):
        return self.model.predict(X)

# Utility Functions
class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.time_window]
            
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded. Max {self.max_calls} calls per {self.time_window} seconds.")
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapped

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_and_cache_data(symbol, time_range):
    """Load and cache stock data with real-time updates"""
    try:
        settings = CONFIG['TIME_RANGES'][time_range]
        
        # For 1D view during market hours, use real-time data
        if time_range == "1D" and is_market_open():
            df = get_real_time_data(symbol)
        else:
            stock = yf.Ticker(symbol)
            df = stock.history(period=settings["period"], 
                             interval=settings["interval"])
        
        if df is None or df.empty:
            return None
            
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Localize to default timezone
        df.index = df.index.tz_convert(CONFIG['DEFAULT_TIMEZONE'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def prepare_data_for_lstm(data, lookback=60):
    """Prepare data for LSTM models"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def calculate_performance_metrics(actual, predicted):
    """Calculate various performance metrics with improved accuracy measures"""
    # Basic metrics
    metrics = {
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted)
    }
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero and handle edge cases
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    metrics['MAPE'] = mape if not np.isinf(mape) else np.nan
    
    # Directional Accuracy
    actual_direction = np.sign(np.diff(actual))
    pred_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    metrics['Directional_Accuracy'] = directional_accuracy
    
    # Price Accuracy (within different tolerance levels)
    tolerances = {
        '1%': 0.01
    }
    
    for label, tolerance in tolerances.items():
        relative_error = np.abs((actual - predicted) / actual)
        within_tolerance = relative_error <= tolerance
        metrics[f'Price_Accuracy_{label}'] = np.mean(within_tolerance) * 100
    
    # R-squared (Coefficient of Determination)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2_Score'] = r2
    
    return metrics

# Chart Creation Functions
def clean_market_data(df):
    """Clean market data by removing non-trading hours and handling gaps"""
    df = df.copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # For Indian market (NSE), trading hours are 9:15 AM to 3:30 PM IST
    market_start = pd.Timestamp('09:15:00').time()
    market_end = pd.Timestamp('15:30:00').time()
    
    # Filter for trading hours
    mask = (df.index.time >= market_start) & (df.index.time <= market_end)
    df = df[mask]
    
    # Remove weekends
    df = df[df.index.dayofweek < 5]  # 0-4 represents Monday-Friday
    
    return df

def get_market_hours():
    """Get market opening and closing hours in IST"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
        
    open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
    return open_time, close_time

def is_market_open():
    """Check if Indian market (NSE) is currently open"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
        
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if current_time.weekday() > 4:
        return False
        
    # Get market hours
    open_time, close_time = get_market_hours()
        
    # Check if current time is within market hours
    return open_time <= current_time <= close_time

def create_realtime_chart(df, show_indicators=False):
    """Create real-time price chart with continuous trading hours"""
    # Clean the data first
    df = clean_market_data(df)
    
    fig = go.Figure()
    
    # Main price candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    if show_indicators:
        # Add Moving Averages
        for ma in ['MA20', 'MA50']:
            if ma in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(width=1)
                ))
        
        # Add Bollinger Bands
        if 'BB_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_upper'],
                name='BB Upper',
                line=dict(dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_lower'],
                name='BB Lower',
                line=dict(dash='dash'),
                fill='tonexty'
            ))
    
    # Update layout with custom axis formatting
    fig.update_layout(
        height=CONFIG['CHART_HEIGHT'],
        xaxis_rangeslider_visible=False,
        title="Real-time Stock Price",
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour"),  # hide non-trading hours
            ]
        )
    )
    
    return fig

def get_real_time_data(symbol, interval='1m'):
    """Get the most recent data possible"""
    try:
        # Get data from yfinance
        stock = yf.Ticker(symbol)
            
        if is_market_open():
            # During market hours: get last 1 hour of 1-minute data
            data = stock.history(period='1h', interval='1m')
        else:
            # Outside market hours: get last available data
            data = stock.history(period='1d', interval=interval)
            
        if data.empty:
            st.warning("No recent data available")
            return None
                
        # Get the last quote
        last_quote = stock.fast_info
            
        # Update the last row with the most recent quote if available
        if hasattr(last_quote, 'last_price') and last_quote.last_price is not None:
            data.iloc[-1]['Close'] = last_quote.last_price
                
        return data
        
    except Exception as e:
        st.error(f"Error fetching real-time data: {str(e)}")
        return None

def create_prediction_chart(df, predictions):
    """Create prediction chart with continuous trading hours"""
    # Clean the data first
    df = clean_market_data(df)
    
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Predicted prices
    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions.flatten(),
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout with custom axis formatting
    fig.update_layout(
        height=CONFIG['CHART_HEIGHT'],
        title="Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour"),  # hide non-trading hours
            ]
        )
    )
    
    return fig

def make_realtime_predictions(model, current_data, n_future=5):
    """Make predictions for current and next n candlesticks"""
    try:
        # Prepare data for prediction
        lookback = CONFIG['LSTM_MODELS'][model]['lookback']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(current_data['Close'].values.reshape(-1, 1))
        
        # Prepare input sequence
        X_pred = scaled_data[-lookback:].reshape(1, lookback, 1)
        
        # Initialize LSTM model
        lstm_model = LSTMModel(ModelType(model), lookback=lookback)
        
        # Make predictions
        predictions = []
        current_sequence = X_pred.copy()
        
        for _ in range(n_future + 1):  # +1 for current candlestick
            next_pred = lstm_model.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions
        
    except Exception as e:
        st.error(f"Error in real-time prediction: {str(e)}")
        return None

def create_combined_chart(df, real_time_data, predictions):
    """Create combined chart with continuous trading hours"""
    # Clean the data
    df = clean_market_data(df)
    real_time_data = clean_market_data(real_time_data)
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Real-time prices
    fig.add_trace(go.Scatter(
        x=real_time_data.index,
        y=real_time_data['Close'],
        name='Real-time Price',
        line=dict(color='green')
    ))
    
    # Predicted prices
    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions.flatten(),
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout with custom axis formatting
    fig.update_layout(
        height=CONFIG['CHART_HEIGHT'],
        title="Combined Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[15.5, 9.25], pattern="hour"),  # hide non-trading hours
            ]
        )
    )
    return fig

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .metric-container {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .prediction-button {
            background-color: #ff4b4b;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def display_metrics(df, predictions=None):
    """Display key metrics in the dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Current price metrics
    latest = df.iloc[-1]
    earliest = df.iloc[0]
    price_change = latest['Close'] - earliest['Close']
    price_change_pct = (price_change / earliest['Close']) * 100
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{latest['Close']:,.2f}",
            f"{price_change_pct:+.2f}%"
        )
    
    # Technical indicators
    with col2:
        if 'RSI' in df.columns:
            st.metric("RSI", f"{latest['RSI']:.2f}")
    
    with col3:
        if 'MACD' in df.columns:
            st.metric("MACD", f"{latest['MACD']:.2f}")
    
    # Prediction metrics
    with col4:
        if predictions is not None:
            pred_change = predictions[-1] - latest['Close']
            pred_change_pct = (pred_change / latest['Close']) * 100
            st.metric(
                "Predicted Change",
                f"₹{predictions[-1]:,.2f}",
                f"{pred_change_pct:+.2f}%"
            )

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Stock Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Add custom CSS for market status
    st.markdown("""
        <style>
        .market-status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .market-status-open {
            background-color: #ffffff;
            color: black;
        }
        .market-status-closed {
            background-color: #ffffff;
            color: black;
        }
        .time-info {
            background-color: #000000;
            padding: 10px;
            border-radius: 5px;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create header with market status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📈 Enhanced Stock Analysis Dashboard")
    
    with col2:
        if is_market_open():
            india_tz = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(india_tz)
            close_time = current_time.replace(hour=15, minute=30, second=0)
            time_to_close = close_time - current_time
            
            if time_to_close.total_seconds() > 0:
                minutes_to_close = int(time_to_close.total_seconds() / 60)
                hours_to_close = minutes_to_close // 60
                mins_remaining = minutes_to_close % 60
                
                st.markdown(
                    f"""
                    <div class="market-status market-status-open">
                        🟢 Market is Open
                    </div>
                    <div class="time-info">
                        Closes in: {hours_to_close}h {mins_remaining}m
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            india_tz = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(india_tz)
            
            if current_time.weekday() >= 5:  # Weekend
                days_to_monday = (7 - current_time.weekday()) % 7
                next_open = current_time + timedelta(days=days_to_monday)
            else:  # Weekday but market closed
                if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 15):
                    next_open = current_time
                else:
                    next_open = current_time + timedelta(days=1)
            
            next_open = next_open.replace(hour=9, minute=15, second=0)
            time_to_open = next_open - current_time
            
            hours_to_open = int(time_to_open.total_seconds() / 3600)
            mins_to_open = int((time_to_open.total_seconds() % 3600) / 60)
            
            st.markdown(
                f"""
                <div class="market-status market-status-closed">
                    🔴 Market is Closed
                </div>
                <div class="time-info">
                    Opens in: {hours_to_open}h {mins_to_open}m
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Theme selector
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
                <style>
                    :root { --background-color: #1e1e1e; --text-color: #ffffff; }
                </style>
            """, unsafe_allow_html=True)
        
        # Stock selector
        selected_symbol = st.selectbox(
            "Select Stock",
            CONFIG['SYMBOLS']
        )
        
        # Time range selector
        selected_range = st.select_slider(
            "Time Range",
            options=list(CONFIG['TIME_RANGES'].keys())
        )
        
        # LSTM model selector
        selected_model = st.selectbox(
            "Select Prediction Model",
            list(CONFIG['LSTM_MODELS'].keys()),
            format_func=lambda x: CONFIG['LSTM_MODELS'][x]['name']
        )
        
        # Update interval info
        update_interval = CONFIG['TIME_RANGES'][selected_range]["update_interval"]
        st.info(f"Data refreshes every {update_interval} seconds")
    
        # Add the Clear Prediction History button here
        if st.button("Clear Prediction History"):
            if 'rt_prediction_history' in st.session_state:
                del st.session_state.rt_prediction_history
            st.success("Prediction history cleared!")

    # Main content area
    try:
        # Load and validate data
        df = load_and_cache_data(selected_symbol, selected_range)
        if df is None or df.empty:
            st.error("No data available for the selected stock and time range")
            return
        
        # Calculate technical indicators
        df = calculate_all_indicators(df)
        
        # Display metrics
        display_metrics(df)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Real-time", "Technical Analysis", "Predictions", "Comparison Table", "Combined Analysis"])
        
        with tab1:
            # Real-time price chart
            st.plotly_chart(
                create_realtime_chart(df, show_indicators=False),
                use_container_width=True
            )
        
        with tab2:
            # Technical analysis chart
            st.plotly_chart(
                create_realtime_chart(df, show_indicators=True),
                use_container_width=True
            )
        
        with tab3:
            # Prediction section
            if st.button("Run Prediction Model", key="predict_button"):
                with st.spinner("Running predictions..."):
                    try:
                        # Initialize and train LSTM model
                        model = LSTMModel(
                            model_type=ModelType(selected_model),
                            lookback=CONFIG['LSTM_MODELS'][selected_model]['lookback']
                        )
                        
                        # Prepare data for prediction
                        X, y, scaler = prepare_data_for_lstm(
                            df['Close'].values,
                            lookback=CONFIG['LSTM_MODELS'][selected_model]['lookback']
                        )
                        
                        # Train model
                        model.train(
                            X, y,
                            epochs=CONFIG['LSTM_MODELS'][selected_model]['epochs'],
                            batch_size=CONFIG['LSTM_MODELS'][selected_model]['batch_size']
                        )
                        
                        # Make predictions
                        predictions = model.predict(X)
                        predictions = scaler.inverse_transform(predictions)
                        
                        # Calculate and display performance metrics
                        metrics = calculate_performance_metrics(
                            df['Close'].values[-len(predictions):],
                            predictions.flatten()
                        )
                        
                        # Performance
                        if predictions is not None:
                            st.subheader("Model Performance Metrics")
                            
                            # Create three columns for metrics display
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                                st.metric("MAE", f"{metrics['MAE']:.2f}")
                            
                            with col2:
                                st.metric("R² Score", f"{metrics['R2_Score']:.3f}")
                                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            
                            with col3:
                                st.metric("Directional Accuracy", f"{metrics['Directional_Accuracy']:.2f}%")
                                st.metric("Price Accuracy (1%)", f"{metrics['Price_Accuracy_1%']:.2f}%")
                            
                            # Add explanation of metrics
                            # with st.expander("Metrics Explanation"):
                                st.markdown("""
                                - **Directional Accuracy**: Percentage of correct movement predictions
                                - **Price Accuracy**: Percentage of predictions within 1% of actual price
                                """)
                        
                        # Display prediction chart
                        st.plotly_chart(
                            create_prediction_chart(df, predictions),
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
                        st.exception(e)
        
                # In the main() function, update the tabs section:

        with tab4:
            st.subheader("Actual vs Predicted Price Comparison")
            
            if 'predictions' in locals():  # Check if predictions exist
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Date': df.index[-len(predictions):],
                    'Actual Price': df['Close'].values[-len(predictions):].round(2),
                    'Predicted Price': predictions.flatten().round(2),
                    'Difference': (predictions.flatten() - df['Close'].values[-len(predictions):]).round(2),
                    'Error %': (np.abs(predictions.flatten() - df['Close'].values[-len(predictions):]) / 
                            df['Close'].values[-len(predictions):] * 100).round(2)
                })
                
                # Add color coding based on error percentage
                def color_error(val):
                    if isinstance(val, str):  # Skip non-numeric columns
                        return ''
                    
                    # For Error % column
                    if abs(val) < 1:
                        return 'background-color: #2E8B57'
                    elif abs(val) < 2:
                        return 'background-color: #DAA520'
                    elif abs(val) < 5:
                        return 'background-color: #CD5C5C'
                    else:
                        return 'background-color: #8B0000'

                # Style the dataframe
                styled_df = comparison_df.style.applymap(
                    color_error, 
                    subset=['Error %']
                ).format({
                    'Actual Price': '₹{:.2f}',
                    'Predicted Price': '₹{:.2f}',
                    'Difference': '₹{:.2f}',
                    'Error %': '{:.2f}%'
                })

                # Add download button
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Comparison Data",
                    data=csv,
                    file_name="price_predictions_comparison.csv",
                    mime="text/csv",
                )

                # Display the table with pagination
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )

                # Display summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Average Error",
                        f"₹{abs(comparison_df['Difference']).mean():.2f}",
                        f"{comparison_df['Error %'].mean():.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Max Error",
                        f"₹{abs(comparison_df['Difference']).max():.2f}",
                        f"{comparison_df['Error %'].max():.2f}%"
                    )
                
                with col3:
                    accurate_predictions = (comparison_df['Error %'] < 5).sum()
                    accuracy_rate = (accurate_predictions / len(comparison_df)) * 100
                    st.metric(
                        "Predictions within 5%",
                        f"{accurate_predictions}/{len(comparison_df)}",
                        f"{accuracy_rate:.1f}%"
                    )

                # Add error distribution chart
                st.subheader("Error Distribution")
                error_fig = go.Figure()
                error_fig.add_trace(go.Histogram(
                    x=comparison_df['Error %'],
                    nbinsx=20,
                    name='Error Distribution'
                ))
                error_fig.update_layout(
                    title="Distribution of Prediction Errors",
                    xaxis_title="Error Percentage",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(error_fig, use_container_width=True)

            else:
                st.info("Run the prediction model to see the comparison table.")

        with tab5:
            st.subheader("Real-time Analysis and Predictions")
            
            # Check if market is open
            market_open = is_market_open()
            
            if market_open:
                st.success("🟢 Market is Open - Real-time updates active")
                
                # Create placeholders
                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Initialize session state for tracking updates and predictions
                if 'last_update' not in st.session_state:
                    st.session_state.last_update = datetime.now(pytz.timezone('Asia/Kolkata'))
                
                if 'rt_prediction_history' not in st.session_state:
                    st.session_state.rt_prediction_history = pd.DataFrame(
                        columns=['Timestamp', 'Actual Close', 'Predicted Close', 'Error %', 'Status', 
                                'Next_1', 'Next_2', 'Next_3', 'Next_4', 'Next_5']
                    )
                
                # Get real-time data
                current_data = get_real_time_data(selected_symbol)
                
                if current_data is not None and not current_data.empty:
                    # Make real-time predictions
                    rt_predictions = make_realtime_predictions(
                        selected_model, 
                        current_data, 
                        n_future=5
                    )
                    
                    if rt_predictions is not None:
                        # Create real-time figure
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            subplot_titles=('Price Analysis with Real-time Predictions', 'Volume'),
                            row_heights=[0.7, 0.3]
                        )

                        # Add candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=current_data.index,
                                open=current_data['Open'],
                                high=current_data['High'],
                                low=current_data['Low'],
                                close=current_data['Close'],
                                name='OHLC'
                            ),
                            row=1, col=1
                        )

                        # Add predictions
                        pred_times = pd.date_range(
                            start=current_data.index[-1],
                            periods=len(rt_predictions),
                            freq=CONFIG['TIME_RANGES']["1D"]["interval"]
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pred_times,
                                y=rt_predictions.flatten(),
                                name='Predictions',
                                line=dict(color='red', dash='dash', width=2),
                                mode='lines+markers'
                            ),
                            row=1, col=1
                        )

                        # Add volume bars
                        fig.add_trace(
                            go.Bar(
                                x=current_data.index,
                                y=current_data['Volume'],
                                name='Volume',
                                marker_color='rgba(0,0,255,0.3)'
                            ),
                            row=2, col=1
                        )

                        # Update layout
                        fig.update_layout(
                            height=800,
                            xaxis_rangeslider_visible=False,
                            showlegend=True,
                            title_text=f"Real-time Analysis - Last Update: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')}",
                            xaxis=dict(
                                type='date',
                                tickformat='%H:%M',
                                rangebreaks=[
                                    dict(bounds=["sat", "mon"]),
                                    dict(bounds=[15.5, 9.25], pattern="hour"),
                                ]
                            )
                        )

                        # Display the plot
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Display predictions metrics
                        with metrics_placeholder.container():
                            cols = st.columns(6)
                            
                            # Current price
                            with cols[0]:
                                current_price = current_data['Close'].iloc[-1]
                                st.metric(
                                    "Current",
                                    f"₹{current_price:.2f}"
                                )
                            
                            # Next 5 predictions
                            for i in range(5):
                                with cols[i+1]:
                                    pred_price = rt_predictions[i+1][0]
                                    change = ((pred_price - current_price) / current_price) * 100
                                    st.metric(
                                        f"T+{i+1}",
                                        f"₹{pred_price:.2f}",
                                        f"{change:+.2f}%"
                                    )
                        

                        with st.expander("ℹ️ Understanding Predictions & Analysis"):
                            # Create three columns for better organization
                            col1, col2, col3 = st.columns([1, 1, 1])
                            
                            with col1:
                                st.markdown("""
                                    ### 📊 Prediction Status
                                    - **Open** : Current active prediction
                                    - **Closed** : Completed prediction with actual results
                                """)
                            
                            with col2:
                                st.markdown("""
                                    ### 🎨 Color Coding System
                                    Error percentage determines color:
                                    
                                    - 🟢 **Dark Green**
                                    - Error < 1%
                                    - Highly Accurate
                                    
                                    - 🟡 **Gold**
                                    - Error 1-2%
                                    - Good Accuracy
                                    
                                    - 🟠 **Indian Red**
                                    - Error 2-5%
                                    - Moderate Accuracy
                                    
                                    - 🔴 **Dark Red**
                                    - Error > 5%
                                    - Low Accuracy
                                """)
                            

                        # Real-time prediction tracking table
                        st.subheader("Real-time Prediction Tracking")
                        
                        # Get current candlestick data
                        current_candle = current_data.iloc[-1]
                        current_time = current_candle.name
                        
                        # Check if candlestick has closed
                        if (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                            current_time.to_pydatetime().replace(tzinfo=pytz.timezone('Asia/Kolkata'))).seconds >= 60:
                            
                            # Add new prediction to history
                            new_prediction = pd.DataFrame({
                                'Timestamp': [current_time],
                                'Actual Close': [current_candle['Close']],
                                'Predicted Close': [rt_predictions[0][0]],
                                'Error %': [abs((rt_predictions[0][0] - current_candle['Close']) / 
                                            current_candle['Close'] * 100)],
                                'Status': ['Closed'],
                                'Next_1': [rt_predictions[1][0]],
                                'Next_2': [rt_predictions[2][0]],
                                'Next_3': [rt_predictions[3][0]],
                                'Next_4': [rt_predictions[4][0]],
                                'Next_5': [rt_predictions[5][0]]
                            })
                            
                            st.session_state.rt_prediction_history = pd.concat(
                                [new_prediction, st.session_state.rt_prediction_history]
                            ).reset_index(drop=True)
                        
                        # Add current (open) prediction
                        current_prediction = pd.DataFrame({
                            'Timestamp': [datetime.now(pytz.timezone('Asia/Kolkata'))],
                            'Prediction_Time': [datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')],
                            'Actual Close': [current_candle['Close']],
                            'Predicted Close': [rt_predictions[0][0]],
                            'Error %': [abs((rt_predictions[0][0] - current_candle['Close']) / 
                                        current_candle['Close'] * 100)],
                            'Status': ['Open'],
                            'Next_1': [rt_predictions[1][0]],
                            'Next_2': [rt_predictions[2][0]],
                            'Next_3': [rt_predictions[3][0]],
                            'Next_4': [rt_predictions[4][0]],
                            'Next_5': [rt_predictions[5][0]]
                        })
                        
                        # Display the table with styling
                        if not st.session_state.rt_prediction_history.empty:
                            # Combine current and historical predictions
                            display_df = pd.concat([
                                current_prediction,
                                st.session_state.rt_prediction_history
                            ]).reset_index(drop=True)
                            
                            # Style the dataframe
                            def style_error(val):
                                if isinstance(val, str):
                                    return ''
                                if val < 1:
                                    return 'background-color: #2E8B57; color: white'
                                elif val < 2:
                                    return 'background-color: #DAA520; color: white'
                                elif val < 5:
                                    return 'background-color: #CD5C5C; color: white'
                                else:
                                    return 'background-color: #8B0000; color: white'
                            
                            def style_status(val):
                                if val == 'Open':
                                    return 'background-color: #1b801f; color: white'
                                return ''
                            
                            # Apply styling
                            styled_df = display_df.style.applymap(
                                style_error, 
                                subset=['Error %']
                            ).applymap(
                                style_status,
                                subset=['Status']
                            ).format({
                                'Actual Close': '₹{:.2f}',
                                'Predicted Close': '₹{:.2f}',
                                'Next_1': '₹{:.2f}',
                                'Next_2': '₹{:.2f}',
                                'Next_3': '₹{:.2f}',
                                'Next_4': '₹{:.2f}',
                                'Next_5': '₹{:.2f}',
                                'Error %': '{:.2f}%'
                            })
                            
                            # Add download button
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prediction History",
                                data=csv,
                                file_name="realtime_predictions_history.csv",
                                mime="text/csv",
                            )
                            
                            # Display table
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=400
                            )
                            
                            # Performance metrics
                            st.subheader("Prediction Performance")
                            closed_predictions = st.session_state.rt_prediction_history[
                                st.session_state.rt_prediction_history['Status'] == 'Closed'
                            ]
                            
                            if not closed_predictions.empty:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_error = closed_predictions['Error %'].mean()
                                    st.metric(
                                        "Average Error",
                                        f"{avg_error:.2f}%"
                                    )
                                
                                with col2:
                                    accuracy_1pct = (closed_predictions['Error %'] < 1).mean() * 100
                                    st.metric(
                                        "Accuracy (within 1%)",
                                        f"{accuracy_1pct:.1f}%"
                                    )
                                
                                with col3:
                                    accuracy_5pct = (closed_predictions['Error %'] < 5).mean() * 100
                                    st.metric(
                                        "Accuracy (within 5%)",
                                        f"{accuracy_5pct:.1f}%"
                                    )
                        
                        # Auto-refresh
                        time.sleep(60)  # Update every minute
                        st.rerun()
                        
                else:
                    st.error("Unable to fetch real-time data")
                    
            else:
                st.warning("🔴 Market is Closed - Real-time updates paused")
                st.info("Real-time analysis and predictions will resume when market opens")
        
        # Add auto-refresh toggle with dynamic refresh rate
        with st.sidebar:
            auto_refresh = st.checkbox("Enable auto-refresh", value=CONFIG['DEFAULT_REFRESH'])
            
            if auto_refresh:
                if is_market_open():
                    refresh_interval = CONFIG['TIME_RANGES'][selected_range]["update_interval"]
                    st.info(f"Data refreshes every {refresh_interval} seconds")
                    
                    # Add JavaScript for auto-refresh
                    st.markdown(
                        f"""
                        <script>
                            setTimeout(function(){{
                                window.location.reload();
                            }}, {refresh_interval * 1000});
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Market is closed. Auto-refresh disabled.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
