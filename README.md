📈 Stock Pro: A Multi-Model Price Forecasting Application
An interactive web application built with Streamlit for fetching, analyzing, and forecasting stock prices using a suite of classical and deep learning time-series models.

🌟 Project Overview
Stock Pro is a full-stack web application designed to make time-series forecasting accessible and comparable. It provides a platform where users can select a stock, train four different industry-standard forecasting models on its historical data, and visualize the predictions for any given date.

The project handles the entire data science pipeline: from live data acquisition and robust cleaning to model training, prediction, and interactive visualization, all wrapped in a secure, user-friendly interface.

✨ Core Features
Secure User Authentication: A complete login/registration system using a local SQLite database, with securely hashed and salted passwords.

Multi-Model Forecasting: Simultaneously trains and compares four powerful models:

ARIMA: A classical statistical model for baseline performance.

SARIMA: An extension of ARIMA that handles seasonality.

Prophet: A modern, automated forecasting library developed by Meta.

LSTM: A deep learning (Recurrent Neural Network) model for capturing complex, non-linear patterns.

Robust Data Pipeline: A "Data Fortification Pipeline" that fetches live data from the yfinance API and automatically cleans it by:

Handling API formatting quirks (like multi-level columns).

Making timestamps timezone-naive.

Validating and removing duplicates or missing values.

Interactive UI: A clean, modern interface built with Streamlit that allows for:

Selection from a curated list of popular global stocks or any custom ticker.

Flexible selection of historical data range ("5 Years" to "All Available").

Interactive charts (via Matplotlib) to visualize data and predictions.

Specific Date Prediction: Users can select any valid past or future date to get a prediction, with intelligent handling of non-trading days (weekends and holidays).

Dashboard View	Predictor View

Export to Sheets
🛠️ Tech Stack
Application Framework: Streamlit

Data Handling & Processing: Pandas, NumPy

Data Source: yfinance API

Machine Learning Models:

statsmodels (for ARIMA/SARIMA)

prophet (for Prophet)

tensorflow & scikit-learn (for the Univariate LSTM)

Database: SQLite3 (for User Authentication)

Visualization: Matplotlib

⚙️ How It Works: The Architecture
The application follows a logical, end-to-end workflow:

User Interaction (Streamlit UI): The user registers, logs in, and selects a stock and a historical data period on the Dashboard.

Data Fetching & Fortification: The fetch_prices function calls the yfinance API and runs the raw data through the robust cleaning and validation pipeline.

On-Demand Model Training: When the user clicks "Load & Train," the train_models function takes the clean data, splits it, and trains all four models. The trained models are cached in memory for the session.

Prediction: On the Predictor page, the user selects a date. The predict_values function uses the cached models to generate a forecast for that specific day.

Visualization: The results are displayed in metric cards and plotted on a Matplotlib chart.

🚀 Setup and Installation
To run this project locally, please follow these steps:

1. Prerequisites:

Python 3.9 or higher

pip and venv

2. Clone the Repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
3. Create and Activate a Virtual Environment:

Bash

# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
4. Install Dependencies:
A requirements.txt file is provided for easy installation.

Bash

pip install -r requirements.txt
(If a requirements.txt is not available, install manually: pip install streamlit pandas numpy yfinance statsmodels prophet scikit-learn tensorflow matplotlib)

5. Run the Application:

Bash

streamlit run stock_pro_final.py
The application will open in a new tab in your web browser.

🔮 Future Improvements
This project provides a solid foundation. Future enhancements could include:

Displaying Performance Metrics: Calculating and showing metrics like RMSE and MAE in the UI to allow users to quantitatively compare model performance.

Backtesting Engine: Building a feature to simulate a trading strategy based on the models' predictions to evaluate historical profitability.

Offline Training Pipeline: For a production-level system, models could be trained daily via a separate script and saved, making predictions instantaneous for the user.

More Advanced Models: Exploring more complex models like LSTMs with attention mechanisms or Transformer networks.
