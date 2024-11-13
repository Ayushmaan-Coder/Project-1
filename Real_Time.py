# config.py
from datetime import timedelta

CONFIG = {
    'TIME_RANGES': {
        "1D": {"interval": "1m", "period": "1d", "update_interval": 60},
        "5D": {"interval": "5m", "period": "5d", "update_interval": 300},
        "1M": {"interval": "30m", "period": "1mo", "update_interval": 1800},
        "3M": {"interval": "1h", "period": "3mo", "update_interval": 3600},
        "6M": {"interval": "1d", "period": "6mo", "update_interval": 86400}
    },
    'DEFAULT_TIMEZONE': 'Asia/Kolkata',
    'CHART_HEIGHT': 500,
    'DEFAULT_REFRESH': True,
    'TABLE_PAGE_SIZE': 50,
    'MAX_RECORDS': 1000,
    'CACHE_TTL': 300,  # 5 minutes
    'API_RATE_LIMIT': 5,  # calls per minute
    'SYMBOLS': [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'HDFC.NS', 'KOTAKBANK.NS'
        # Add more symbols as needed
    ]
}

# utils.py
import logging
import time
from functools import wraps
from datetime import datetime
import pandas as pd
import pytz
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def validate_stock_data(df):
    """Validate the structure and content of stock data."""
    if df is None or df.empty:
        return False
    required_columns = ['Time', 'Open Price', 'Close Price']
    return all(col in df.columns for col in required_columns)

def cleanup_historical_data():
    """Clean up historical data to prevent memory issues."""
    if 'historical_data' in st.session_state:
        df = st.session_state.historical_data
        if len(df) > CONFIG['MAX_RECORDS']:
            st.session_state.historical_data = df.tail(CONFIG['MAX_RECORDS'])

# technical_indicators.py
def calculate_moving_average(df, window=20):
    """Calculate moving average for the given window."""
    df[f'MA{window}'] = df['Close Price'].rolling(window=window).mean()
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    delta = df['Close Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['MA20'] = df['Close Price'].rolling(window=window).mean()
    df['BB_upper'] = df['MA20'] + 2 * df['Close Price'].rolling(window=window).std()
    df['BB_lower'] = df['MA20'] - 2 * df['Close Price'].rolling(window=window).std()
    return df

# main.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pytz

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with theme support
def apply_custom_css():
    st.markdown("""
        <style>
        :root {
            --background-color: #ffffff;
            --text-color: #000000;
        }
        [data-theme="dark"] {
            --background-color: #1e1e1e;
            --text-color: #ffffff;
        }
        .stDataFrame {
            font-size: 14px;
        }
        .view-control {
            background-color: var(--background-color);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: var(--background-color);
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
@RateLimiter(max_calls=CONFIG['API_RATE_LIMIT'], time_window=60)
def fetch_stock_data(symbol, time_range):
    """Fetch stock data with caching and rate limiting."""
    try:
        logger.info(f"Fetching data for {symbol} with range {time_range}")
        stock = yf.Ticker(symbol)
        if not stock:
            raise ValueError(f"Invalid symbol: {symbol}")
            
        settings = CONFIG['TIME_RANGES'].get(time_range)
        if not settings:
            raise ValueError(f"Invalid time range: {time_range}")
        
        hist = stock.history(period=settings["period"], interval=settings["interval"])
        
        if hist.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Convert to IST timezone
        ist = pytz.timezone(CONFIG['DEFAULT_TIMEZONE'])
        hist.index = hist.index.tz_convert(ist)
        
        # Format data
        data = []
        for timestamp, row in hist.iterrows():
            data.append({
                "Time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Open Price": round(row["Open"], 2),
                "High Price": round(row["High"], 2),
                "Low Price": round(row["Low"], 2),
                "Close Price": round(row["Close"], 2),
                "Volume": row["Volume"]
            })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators
        df = calculate_moving_average(df, window=20)
        df = calculate_moving_average(df, window=50)
        df = calculate_rsi(df)
        df = calculate_bollinger_bands(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_advanced_chart(df):
    """Create an advanced interactive chart with technical indicators."""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Time'],
        open=df['Open Price'],
        high=df['High Price'],
        low=df['Low Price'],
        close=df['Close Price'],
        name='OHLC'
    ))
    
    # Add Moving Averages
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['MA20'],
            name='MA20',
            line=dict(color='blue', width=1)
        ))
    
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['MA50'],
            name='MA50',
            line=dict(color='orange', width=1)
        ))
    
    # Add Bollinger Bands
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
    
    # Volume subplot
    fig.add_trace(go.Bar(
        x=df['Time'],
        y=df['Volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    # Layout updates
    fig.update_layout(
        title='Advanced Technical Analysis',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        height=CONFIG['CHART_HEIGHT'],
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def display_metrics(df):
    """Display key metrics in a dashboard format."""
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        earliest = df.iloc[0]
        
        # Calculate metrics
        price_change = latest['Close Price'] - earliest['Close Price']
        price_change_pct = (price_change / earliest['Close Price']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"₹{latest['Close Price']:,.2f}",
                f"{price_change_pct:+.2f}%"
            )
            
        with col2:
            st.metric(
                "Volume",
                f"{latest['Volume']:,}",
                f"{((latest['Volume'] - earliest['Volume']) / earliest['Volume'] * 100):+.2f}%"
            )
            
        with col3:
            if 'RSI' in df.columns:
                st.metric("RSI", f"{latest['RSI']:.2f}")
                
        with col4:
            if 'MA20' in df.columns:
                ma_signal = "ABOVE MA20" if latest['Close Price'] > latest['MA20'] else "BELOW MA20"
                st.metric("MA Signal", ma_signal)

def main():
    """Main application function."""
    apply_custom_css()
    
    st.title("📈 Enhanced Stock Dashboard")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Theme selector
        theme = st.selectbox("Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""<script>document.body.setAttribute('data-theme', 'dark');</script>""", 
                       unsafe_allow_html=True)
        
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
        
        # Technical indicators
        st.subheader("Technical Indicators")
        show_ma = st.checkbox("Moving Averages", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        
        update_interval = CONFIG['TIME_RANGES'][selected_range]["update_interval"]
        st.info(f"Data refreshes every {update_interval} seconds")
    
    # Main content area
    should_refresh = (
        st.button("🔄 Refresh Data") or 
        'last_refresh' not in st.session_state or
        selected_symbol != st.session_state.get('current_symbol') or
        selected_range != st.session_state.get('current_range')
    )
    
    if should_refresh:
        with st.spinner('Fetching latest data...'):
            df = fetch_stock_data(selected_symbol, selected_range)
            if df is not None and validate_stock_data(df):
                st.session_state.historical_data = df
                st.session_state.current_symbol = selected_symbol
                st.session_state.current_range = selected_range
                st.session_state.last_refresh = datetime.now()
                cleanup_historical_data()
    
    # Display metrics dashboard
    if 'historical_data' in st.session_state:
        display_metrics(st.session_state.historical_data)
        
        # Chart and table tabs
        tab1, tab2 = st.tabs(["Chart", "Data"])
        
        with tab1:
            st.plotly_chart(
                create_advanced_chart(st.session_state.historical_data),
                use_container_width=True
            )
            
        with tab2:
            st.dataframe(
                st.session_state.historical_data.sort_values('Time', ascending=False),
                use_container_width=True
            )
    
    # Show last update time
    if 'last_refresh' in st.session_state:
        st.markdown(f"*Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Auto-refresh using JavaScript
    if st.checkbox("Enable auto-refresh", value=CONFIG['DEFAULT_REFRESH']):
        st.markdown(
            f"""
            <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {update_interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
