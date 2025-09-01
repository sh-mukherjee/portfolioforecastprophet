import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st

# --- Configuration ---
PORTFOLIO_ASSETS = {
    'VT': 0.60,  # Vanguard Total World Stock ETF (Stocks)
    'BNDW': 0.40   # VVanguard Total World Bond ETF (Bonds)
}
INITIAL_INVESTMENT = 10000

# Date Range for Historical Data (fetch last 5 years)
END_DATE = dt.datetime.now().strftime('%Y-%m-%d')
START_DATE = (dt.datetime.now() - dt.timedelta(days=5*365)).strftime('%Y-%m-%d') # 5 years ago

# Forecasting period
FORECAST_PERIOD_DAYS = 90 # Forecast for the next 90 days

# --- Step 1: Fetch Historical ETF Data ---
@st.cache_data # Cache data fetching for performance
def fetch_historical_data(tickers, start_date, end_date):
    """Fetches historical 'Close' prices for given tickers."""
    st.write(f"Fetching historical data from {start_date} to {end_date} for: {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date)
    if len(tickers) > 1:
        # Filter out rows where all ticker close prices are NaN (e.g., market holidays)
        return data['Close'].dropna(how='all')
    else:
        # Ensure it's a DataFrame and drop NaNs if single ticker
        return data['Close'].to_frame().dropna()

# --- Step 2: Calculate Historical Portfolio Value ---
def calculate_portfolio_value(df_prices, weights, initial_investment):
    """Calculates the daily historical value of the portfolio."""
    portfolio_value = pd.Series(index=df_prices.index, dtype=float)

    # Calculate initial shares for each asset
    initial_shares = {}
    valid_data_found = False
    for ticker, weight in weights.items():
        if ticker in df_prices.columns and not df_prices[ticker].empty:
            first_price = df_prices[ticker].iloc[0]
            if first_price > 0: # Ensure price is valid
                initial_allocation = initial_investment * weight
                initial_shares[ticker] = initial_allocation / first_price
                valid_data_found = True
            else:
                st.warning(f"Warning: First price for {ticker} is zero or negative. Skipping {ticker}.")
                initial_shares[ticker] = 0
        else:
            st.warning(f"Warning: {ticker} not found in fetched data or data is empty. Skipping.")
            initial_shares[ticker] = 0

    if not valid_data_found:
        st.error("Error: No valid data found to calculate initial shares for any asset. Check inputs or data availability.")
        return pd.Series(dtype=float) # Return empty series

    # Calculate daily portfolio value
    for date in df_prices.index:
        daily_value = 0
        for ticker, shares in initial_shares.items():
            if ticker in df_prices.columns:
                # Use .get() with a default to handle potential missing values for a specific day
                price = df_prices.loc[date, ticker] if date in df_prices.index else 0
                daily_value += shares * price
        portfolio_value.loc[date] = daily_value

    return portfolio_value.dropna() # Drop any NaN values that might result from missing data points


# --- Step 3: Prepare Data for Prophet ---
def prepare_prophet_data(portfolio_series):
    """Converts a pandas Series to Prophet's required DataFrame format."""
    df_prophet = portfolio_series.reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    return df_prophet

# --- Step 4: Train Prophet Model ---
@st.cache_resource # Cache the model training
def train_prophet_model(df_prophet, growth='linear', seasonality_mode='additive'):
    """Trains a Prophet model."""
    st.info("Training Prophet model...")
    m = Prophet(
        growth=growth, # 'linear' or 'logistic' (for bounded growth)
        seasonality_mode=seasonality_mode, # 'additive' or 'multiplicative'
        daily_seasonality=False, # Often not significant for daily stock data
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    m.add_country_holidays(country_name='US')
    m.fit(df_prophet)
    st.success("Prophet model trained successfully.")
    return m

# --- Step 5: Make Forecasts ---
def make_forecast(model, periods):
    """Generates future dates and makes predictions."""
    st.info(f"Generating forecasts for the next {periods} days...")
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    st.success("Forecasts generated.")
    return forecast

# --- Step 6: Visualize Results with Plotly Express ---
def visualize_forecast(model, forecast, historical_df, title="Portfolio Value Forecast"):
    """Plots the historical data, forecast, and components using Plotly."""

    # Plotting the main forecast
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines',
        name='Historical Value',
        line=dict(color='blue')
    ))

    # Forecasted values
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))

    # Uncertainty interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(width=0),
        name='Uncertainty Interval',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified" # Shows all values for a given date on hover
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plotting forecast components manually with Plotly Express
    st.subheader("Forecast Components")

    # Trend
    fig_trend = px.line(forecast, x='ds', y='trend', title='Trend Component')
    st.plotly_chart(fig_trend, use_container_width=True)

    # Weekly Seasonality
    # Prophet's forecast dataframe includes 'weekly' and 'yearly' columns
    # but these are generally only valid within the historical period.
    # For future, they are projected. We can plot them directly.
    # To get a smoother weekly plot, we might use the model's internal data
    # or just plot directly from forecast.
    # Let's plot the seasonality for the entire forecast period for consistency

    fig_weekly = px.line(forecast, x='ds', y='weekly', title='Weekly Seasonality')
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Yearly Seasonality
    fig_yearly = px.line(forecast, x='ds', y='yearly', title='Yearly Seasonality')
    st.plotly_chart(fig_yearly, use_container_width=True)

    # Holidays (if added and significant, will appear as a separate column 'holidays' or sum into 'extra_regressors')
    # If you added custom holidays, Prophet might create a 'holidays' column
    # or include them in 'extra_regressors'.
    # We can check if specific holiday columns exist in the forecast DataFrame.
    # For built-in Prophet holidays, they are usually included in the 'daily', 'weekly', 'yearly' components
    # or aggregated into 'extra_regressors_additive'.
    # Let's check for 'holidays' component if it exists.
    if 'holidays' in forecast.columns:
        fig_holidays = px.line(forecast, x='ds', y='holidays', title='Holiday Effect')
        st.plotly_chart(fig_holidays, use_container_width=True)
    elif 'extra_regressors_additive' in forecast.columns:
         fig_extra_reg = px.line(forecast, x='ds', y='extra_regressors_additive', title='Extra Regressors (Holidays/Events)')
         st.plotly_chart(fig_extra_reg, use_container_width=True)


# --- Step 7: Evaluate Model (Optional but Recommended) ---
def evaluate_model(df_prophet, model_placeholder, horizon_days=30):
    """Evaluates the model performance on a historical test set."""
    st.info(f"Evaluating model performance on the last {horizon_days} days...")
    # Split data into train and test
    train_size = len(df_prophet) - horizon_days
    if train_size <= 0:
        st.warning("Not enough data to create a test set of the specified horizon.")
        return None

    train_df = df_prophet.iloc[:train_size]
    test_df = df_prophet.iloc[train_size:]

    # Re-initialize a Prophet model for evaluation
    m_eval = Prophet(
        growth='linear',
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    m_eval.add_country_holidays(country_name='US')
    m_eval.fit(train_df)

    future_eval = m_eval.make_future_dataframe(periods=horizon_days)
    forecast_eval = m_eval.predict(future_eval)

    # Merge actual and predicted values for evaluation
    comparison_df = pd.merge(test_df, forecast_eval[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')

    if comparison_df.empty:
        st.warning("No overlapping dates for evaluation. Adjust test horizon or data range.")
        return None

    mae = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
    rmse = np.sqrt(mean_squared_error(comparison_df['y'], comparison_df['yhat']))
    mape = np.mean(np.abs((comparison_df['y'] - comparison_df['yhat']) / comparison_df['y'])) * 100

    st.subheader(f"Model Evaluation (Last {horizon_days} Days)")
    st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")

    # Plot actual vs. predicted for evaluation period with Plotly Express
    fig_eval = go.Figure()

    fig_eval.add_trace(go.Scatter(
        x=train_df['ds'],
        y=train_df['y'],
        mode='lines',
        name='Historical Training Data',
        line=dict(color='blue')
    ))
    fig_eval.add_trace(go.Scatter(
        x=test_df['ds'],
        y=test_df['y'],
        mode='lines',
        name='Actual Test Data',
        line=dict(color='orange')
    ))
    fig_eval.add_trace(go.Scatter(
        x=comparison_df['ds'], # Use comparison_df for shared dates
        y=comparison_df['yhat'],
        mode='lines',
        name='Predicted Test Data',
        line=dict(color='green', dash='dash')
    ))
    fig_eval.add_trace(go.Scatter(
        x=comparison_df['ds'],
        y=comparison_df['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_eval.add_trace(go.Scatter(
        x=comparison_df['ds'],
        y=comparison_df['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(width=0),
        name='Prediction Interval',
        hoverinfo='skip'
    ))

    fig_eval.update_layout(
        title=f"Model Evaluation: Actual vs. Predicted Portfolio Value (Last {horizon_days} Days)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    return {'mae': mae, 'rmse': rmse, 'mape': mape}

# --- Main Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Portfolio Performance Forecaster")
st.title("💰 Portfolio Performance Forecaster with Prophet (Plotly Edition)")
st.write("Forecast the future value of a hypothetical portfolio of ETFs using Meta's Prophet, visualized with Plotly.")

# Sidebar for user inputs
st.sidebar.header("Configuration")
initial_investment_input = st.sidebar.number_input(
    "Initial Investment ($):",
    min_value=1000,
    max_value=1000000,
    value=INITIAL_INVESTMENT,
    step=1000
)

st.sidebar.subheader("Portfolio Allocation:")
vt_weight = st.sidebar.slider("SPY Weight (%):", 0, 100, int(PORTFOLIO_ASSETS['VT']*100))
bnd_weight = 100 - vt_weight
st.sidebar.write(f"BND Weight (%): {bnd_weight}")

# Update weights based on slider
current_portfolio_assets = {
    'VT': vt_weight / 100.0,
    'BND': bnd_weight / 100.0
}

forecast_period_input = st.sidebar.slider(
    "Forecast Period (Days):",
    min_value=30,
    max_value=365,
    value=FORECAST_PERIOD_DAYS,
    step=30
)

st.sidebar.markdown("---")
st.sidebar.info("Data fetched from Yahoo Finance for the last 5 years.")

# Main app logic
if st.button("Generate Forecast"):
    st.subheader("Processing Data...")

    tickers = list(current_portfolio_assets.keys())

    # 1. Fetch data
    df_prices = fetch_historical_data(tickers, START_DATE, END_DATE)

    if df_prices.empty:
        st.error("Error: No historical data fetched. Please check ETF tickers or date range.")
    else:
        # 2. Calculate portfolio value
        portfolio_series = calculate_portfolio_value(df_prices, current_portfolio_assets, initial_investment_input)

        if portfolio_series.empty:
            st.error("Error: Portfolio value calculation resulted in an empty series. Adjust initial investment or asset weights.")
        else:
            # 3. Prepare data for Prophet
            df_prophet = prepare_prophet_data(portfolio_series)

            # 4. Evaluate the model on a portion of the historical data
            st.markdown("---")
            evaluate_model(df_prophet, None, horizon_days=30) # 'None' is a placeholder for model_placeholder

            # 5. Train the final Prophet model on ALL historical data
            st.markdown("---")
            with st.spinner("Training Prophet model on full data..."):
                m = train_prophet_model(df_prophet)
            st.success("Model trained!")

            # 6. Make future forecasts
            with st.spinner(f"Generating forecasts for the next {forecast_period_input} days..."):
                forecast = make_forecast(m, forecast_period_input)
            st.success("Forecasts generated!")

            # 7. Visualize results
            st.markdown("---")
            st.subheader("Future Portfolio Value Forecast")
            visualize_forecast(m, forecast, df_prophet, "Portfolio Value Forecast") # Use the custom Plotly visualization

            st.subheader("Raw Forecast Data")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
