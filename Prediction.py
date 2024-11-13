import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, timedelta
from sklearn.model_selection import train_test_split

# Constants
START = (date.today() - timedelta(days=10*365)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")
SENSEX_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
    'BHARTIARTL.NS', 'SBIN.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS',
    'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS',
    'TITAN.NS', 'BAJAJFINSV.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'TECHM.NS', 'POWERGRID.NS',
    'NTPC.NS', 'M&M.NS', 'ONGC.NS', 'HDFCLIFE.NS', 'TATASTEEL.NS'
]

st.title('Advanced Sensex Stock Forecast App')

selected_stock = st.selectbox('Select Sensex stock for prediction', SENSEX_TICKERS)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def add_technical_indicators(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df['ATR'] = calculate_atr(df)
    df['OBV'] = calculate_obv(df)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
    df['Stochastic_K'], df['Stochastic_D'] = calculate_stochastic_oscillator(df)
    df['ADX'] = calculate_adx(df)
    df['CCI'] = calculate_cci(df)
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window).mean()
    return k, d

def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(df['High'] - df['Low'])
    tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
    tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (period - 1)) + dx) / period
    adx_smooth = adx.ewm(alpha=1/period).mean()
    return adx_smooth

def calculate_cci(df, period=20):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = (TP - TP.rolling(period).mean()) / (0.015 * TP.rolling(period).std())
    return CCI

data = load_data(selected_stock)
data = add_technical_indicators(data)

# Remove rows with NaN values
data.dropna(inplace=True)

st.subheader('Raw data with Technical Indicators')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name="MA50"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name="MA200"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Upper'], name="Bollinger Upper"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Bollinger_Lower'], name="Bollinger Lower"))
    fig.layout.update(title_text='Time Series data with Indicators', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for bi-directional model
n_past = 60  # Number of past time steps
n_future = 30  # Number of future time steps to predict

def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data[i:(i + n_past)])
        y.append(data[(i + n_past):(i + n_past + n_future), 0])
    return np.array(X), np.array(y)

# Use only the 'Close' price for scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

X, y = create_sequences(scaled_data, n_past, n_future)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the bi-directional LSTM model
def create_bidirectional_lstm_model(input_shape, output_length):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_length)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

model = create_bidirectional_lstm_model((n_past, 1), n_future)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Reshape predictions and test data
predictions_reshaped = predictions[:,-1].reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Inverse transform predictions and actual values
predictions_inv = scaler.inverse_transform(predictions_reshaped)
y_test_inv = scaler.inverse_transform(y_test[:, -1].reshape(-1, 1))

display_df = data.tail(len(predictions_inv)).copy()
display_df['Actual_Close'] = y_test_inv
display_df['Predicted_Close'] = predictions_inv
display_df = display_df[['Open', 'High', 'Low', 'Close', 'Actual_Close', 'Predicted_Close']]

float_columns = display_df.select_dtypes(include=['float64']).columns
display_df[float_columns] = display_df[float_columns].round(2)

st.subheader('Actual vs Predicted Prices')
st.dataframe(display_df)

# Calculate performance metrics
mse = mean_squared_error(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)

def calculate_directional_accuracy(y_true, y_pred):
    y_true_direction = np.sign(np.diff(y_true.flatten()))
    y_pred_direction = np.sign(np.diff(y_pred.flatten()))
    directional_accuracy = np.mean(y_true_direction == y_pred_direction)
    return directional_accuracy

def calculate_price_accuracy(y_true, y_pred, tolerance_percentage=0.08):
    tolerance = y_true * tolerance_percentage
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    price_accuracy = np.mean(within_tolerance)
    return price_accuracy

directional_accuracy = calculate_directional_accuracy(y_test_inv, predictions_inv)
price_accuracy = calculate_price_accuracy(y_test_inv, predictions_inv)

st.subheader('Model Performance Metrics')
st.write(f'Mean Squared Error: {mse:.4f}')
st.write(f'Mean Absolute Error: {mae:.4f}')
st.write(f'Directional Accuracy: {directional_accuracy:.2%}')
st.write(f'Price Accuracy: {price_accuracy:.2%}')

# Visualize results
st.subheader('Bi-directional Forecast Results')
fig = go.Figure()

# Plot actual prices
fig.add_trace(go.Scatter(x=data.index[-len(y_test_inv):], y=y_test_inv.flatten(),
                         mode='lines', name='Actual Price'))

# Plot predicted prices
fig.add_trace(go.Scatter(x=data.index[-len(predictions_inv):], y=predictions_inv.flatten(),
                         mode='lines', name='Predicted Price'))

fig.update_layout(title='Bi-directional Stock Price Forecast',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

st.plotly_chart(fig)

# Simulated Trading Scenario
initial_balance = 100000
balance = initial_balance
shares = 0
transaction_cost = 0.001  # 0.1% transaction cost

# After making predictions
predictions = model.predict(X_test)

# Reshape predictions and prepare for inverse transformation
predictions_reshaped = predictions[:, -1].reshape(-1, 1)  # Take only the last predicted value for each sequence

# Inverse transform
predictions_inv = scaler.inverse_transform(predictions_reshaped)

# Prepare the valid DataFrame
valid = data.tail(len(predictions_inv))
valid['Predicted_Price'] = predictions_inv.flatten()  # Use the flattened predictions

for i in range(1, len(valid)):
    if valid['Predicted_Price'].iloc[i] > valid['Close'].iloc[i-1] and balance > 0:  # Buy signal
        shares_to_buy = balance // (valid['Close'].iloc[i] * (1 + transaction_cost))
        cost = shares_to_buy * valid['Close'].iloc[i] * (1 + transaction_cost)
        shares += shares_to_buy
        balance -= cost
    elif valid['Predicted_Price'].iloc[i] < valid['Close'].iloc[i-1] and shares > 0:  # Sell signal
        revenue = shares * valid['Close'].iloc[i] * (1 - transaction_cost)
        balance += revenue
        shares = 0

final_balance = balance + shares * valid['Close'].iloc[-1]
returns = (final_balance - initial_balance) / initial_balance * 100

st.subheader('Simulated Trading Results')
st.write(f'Initial Balance: ${initial_balance:.2f}')
st.write(f'Final Balance: ${final_balance:.2f}')
st.write(f'Returns: {returns:.2f}%')
st.write(f'Annualized Returns: {(returns / (len(valid) / 252)):.2f}%')
