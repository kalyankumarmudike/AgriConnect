import pandas as pd
from datetime import datetime, timedelta
from joblib import dump, load
import os
import requests
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from flask import Flask, jsonify, request
from flask_cors import CORS 
# ... [keep your existing locations list and helper functions] ...


locations = [
    "Devarkonda(Mallepalli)", "Devarakonda", "Devarkonda(Dindi)", "Huzzurabad",
    "Suryapeta", "Tirumalagiri", "Kodad", "Huzumnagar(Garidepally)", "Gangadhara",
    "Huzurnagar(Matampally)", "Wyra", "Huzurnagar", "Neredcherla", "Kallur",
    "Dammapet", "Burgampadu", "Bhongir", "Dharmaram", "Thungathurthy", "Bhadrachalam",
    "Manakodur", "Alampur", "Vemulawada", "Charla", "Kothagudem", "Gadwal(Lezza)",
    "Manthani", "Jagtial", "Narayanpet", "Devarakadra"
]




def create_smart_lags(df):
    # Sort by date first
    df = df.sort_values(['district', 'market', 'variety', 'grade', 'price_date'])

    # Group by market-product combination
    grouped = df.groupby(['district', 'market', 'variety', 'grade'])

    # Calculate smart lags
    for lag in [1, 2, 3, 7]:  # 1-day, 2-day, etc. lags
        df[f'max_price_lag_{lag}'] = grouped['max_price'].shift(lag)
        df[f'min_price_lag_{lag}'] = grouped['min_price'].shift(lag)
        df[f'modal_price_lag_{lag}'] = grouped['modal_price'].shift(lag)

    return df

def calculate_rolling_features(df):
    """Calculate rolling features for the updated dataset"""
    window_sizes = [3]  # 3-day, 1-week, 2-week windows
    for window in window_sizes:
        df[f'max_price_rolling_mean_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['max_price'].transform(lambda x: x.rolling(window).mean())
        df[f'max_price_rolling_std_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['max_price'].transform(lambda x: x.rolling(window).std())
        df[f'min_price_rolling_mean_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['min_price'].transform(lambda x: x.rolling(window).mean())
        df[f'min_price_rolling_std_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['min_price'].transform(lambda x: x.rolling(window).std())
        df[f'modal_price_rolling_mean_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['modal_price'].transform(lambda x: x.rolling(window).mean())
        df[f'modal_price_rolling_std_{window}'] = df.groupby(['district', 'market', 'variety', 'grade'])['modal_price'].transform(lambda x: x.rolling(window).std())
    
    return df.dropna()

def update_model_with_csv_data():
    """Daily update workflow using CSV files"""
    try:
        # 1. Load today's new data
        response = requests.get("http://127.0.0.1:8000/scrape")
        data = []
        if response.status_code == 200:
            data = response.json()
        else:
            print("Error:", response.status_code)
            return None, None

        today_data = pd.DataFrame(data)
        today_data['price_date'] = pd.to_datetime(today_data['price_date'])
        today_data = today_data[today_data["market"].isin(locations)]

        # Debug: Print markets found in today's data
        print("Markets in today's data:", today_data['market'].unique())

        historical_data = pd.read_csv('data/agmarknet_data.csv')
        historical_data['price_date'] = pd.to_datetime(historical_data['price_date'])

        # Debug: Print markets in historical data
        # print("Markets in historical data:", historical_data['market'].unique())

        unique_Values = len(today_data['market'].unique())
        # if unique_Values < 37 :
        #     model = load('models/current_model.joblib')
        #     return model, historical_data
        
        updated_data = pd.concat([historical_data, today_data])
        updated_data = updated_data.sort_values(['market', 'price_date'])

        

        updated_data.to_csv('data/agmarknet_data.csv', index=False)
        updated_data = create_smart_lags(updated_data)
        updated_data = calculate_rolling_features(updated_data)
        
        # Add temporal features
        updated_data['day_of_week'] = updated_data['price_date'].dt.dayofweek
        updated_data['day_of_month'] = updated_data['price_date'].dt.day
        updated_data['month'] = updated_data['price_date'].dt.month
        updated_data['year'] = updated_data['price_date'].dt.year
        

        features = ['district', 'market', 'variety', 'grade', 'day_of_week', 
                   'day_of_month', 'month', 'year'] + \
                  [col for col in updated_data.columns if 'lag_' in col or 'rolling_' in col]
        targets = ['max_price', 'min_price', 'modal_price']


        X_train = updated_data[features]
        y_train = updated_data[targets]


        preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['district', 'market', 'variety', 'grade']),
            ('num', StandardScaler(), [col for col in features if col not in ['district', 'market', 'variety', 'grade']])
        ])

        # Model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(GradientBoostingRegressor()))
        ])

        model.fit(X_train,y_train)
        
        # Save artifacts
        updated_data.drop(columns=["day_of_week", "day_of_month", "month", "year"], inplace=True)
        dump(model, 'models/current_model.joblib')
        dump(model, f'models/model_versions/model_{datetime.now().date()}.joblib')
        
        return model, updated_data
    
    except Exception as e:
        print(f"Error in update_model_with_csv_data: {str(e)}")
        return None, None
    


def prepare_tomorrows_input(market, district, variety, grade, current_data):
    """
    Prepare input data for tomorrow's prediction

    Args:
        market: Name of the market (e.g., "Devarkonda(Mallepalli)")
        district: District name
        variety: Paddy variety
        grade: Grade of paddy
        current_data: Your complete DataFrame (df_with_lags)
    """
    # Get dates
    today = datetime.today()
    tomorrow = today + timedelta(days=1)

    # Get the last 7 days of data for this specific market-product combination
    market_data = current_data[
        (current_data['market'] == market) &
        (current_data['district'] == district) &
        (current_data['variety'] == variety) &
        (current_data['grade'] == grade)
    ].sort_values('price_date').tail(7)

    if len(market_data) == 0:
        raise ValueError(f"No historical data found for market: {market}, district: {district}")

    # Get the most recent record
    latest = market_data.iloc[-1]

    # Create input features
    input_data = {
        'district': district,
        'market': market,
        'variety': variety,
        'grade': grade,
        'day_of_week': tomorrow.weekday(),
        'day_of_month': tomorrow.day,
        'month': tomorrow.month,
        'year': tomorrow.year,
        # Lag features (using latest available data)
        'max_price_lag_1': latest['max_price'],
        'min_price_lag_1': latest['min_price'],
        'modal_price_lag_1': latest['modal_price'],
        'max_price_lag_2': latest['max_price_lag_1'],
        'min_price_lag_2': latest['min_price_lag_1'],
        'modal_price_lag_2': latest['modal_price_lag_1'],
        'max_price_lag_3': latest['max_price_lag_2'],
        'min_price_lag_3': latest['min_price_lag_2'],
        'modal_price_lag_3': latest['modal_price_lag_2'],
        'max_price_lag_7': latest['max_price_lag_3'],  # Adjust if you have exact 7-day lag
        'min_price_lag_7': latest['min_price_lag_3'],
        'modal_price_lag_7': latest['modal_price_lag_3'],
        # Rolling features (recalculate based on available data)
        'max_price_rolling_mean_3': market_data['max_price'].tail(3).mean(),
        'max_price_rolling_std_3': market_data['max_price'].tail(3).std(),
        'min_price_rolling_mean_3': market_data['min_price'].tail(3).mean(),
        'min_price_rolling_std_3': market_data['min_price'].tail(3).std(),
        'modal_price_rolling_mean_3': market_data['modal_price'].tail(3).mean(),
        'modal_price_rolling_std_3': market_data['modal_price'].tail(3).std(),
        # Add all other features you used in training
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    features = ['district', 'market', 'variety', 'grade', 'day_of_week', 
                   'day_of_month', 'month', 'year'] + \
                  [col for col in input_df.columns if 'lag_' in col or 'rolling_' in col]

    # Ensure all columns are in same order as training
    input_df = input_df[features]  # 'features' should be your original feature list

    return input_df

def predict_tomorrows_prices(model, market, district, variety, grade, current_data):
    """
    Predict tomorrow's prices for a specific market and product

    Args:
        model: Your trained sklearn model
        market: Market name (e.g., "Devarkonda(Mallepalli)")
        district: District name
        variety: Paddy variety
        grade: Grade
        current_data: Your complete DataFrame (df_with_lags)
    """
    # Get dates
    today = datetime.today()
    tomorrow = today + timedelta(days=1)

    # Prepare input data
    tomorrows_input = prepare_tomorrows_input(
        market=market,
        district=district,
        variety=variety,
        grade=grade,
        current_data=current_data
    )

    # Make prediction
    predicted_prices = model.predict(tomorrows_input)

    # Format output
    print(f"\nPredicted prices for {market} on {tomorrow.strftime('%Y-%m-%d')}:")
    print(f"Max Price: {predicted_prices[0][0]:.2f}")
    print(f"Min Price: {predicted_prices[0][1]:.2f}")
    print(f"Modal Price: {predicted_prices[0][2]:.2f}")

    return {
        'market': market,
        'date': tomorrow.strftime('%Y-%m-%d'),
        'max_price': predicted_prices[0][0],
        'min_price': predicted_prices[0][1],
        'modal_price': predicted_prices[0][2]
    }

# Example usage:
# market = "Dharmaram"
# district = "Karimnagar"  # Replace with actual district
# variety = "1001"  # Replace with actual variety
# grade = "FAQ"   # Replace with actual grade

district = "Nalgonda"
market = "Tirumalagiri"  # Note: This should be the market name exactly as in your data
variety = "I.R. 64"
grade = "FAQ"


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask API on port 9500!"})

@app.route('/model_prediction', methods=['GET'])
def model_prediction():
    model ,data = update_model_with_csv_data()  # get query parameter ?input=...
     
    time.sleep(10)
    if data is None or data.empty:
        return jsonify({"error": "No input provided"}), 400

    prediction = predict_tomorrows_prices(
        model=model,
        market=market,
        district=district,
        variety=variety,
        grade=grade,
        current_data=data
        )
    time.sleep(10)
    return jsonify(prediction)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'preflight allowed'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    try:
        # Get input data from request
        input_data = request.get_json()
        
        # Validate required fields
        print(input_data)
        required_fields = ['district', 'market', 'variety', 'grade']
        if not all(field in input_data for field in required_fields):
            return jsonify({"error": "Missing required fields. Please provide district, market, variety, and grade."}), 400
        
        district = input_data['district']
        market = input_data['market']
        variety = input_data['variety']
        grade = input_data['grade']
        
        # Load model and data
        model, data = update_model_with_csv_data()
        time.sleep(10)
        if data is None or data.empty:
            return jsonify({"error": "No historical data available for prediction"}), 500
        
        # Make prediction
        prediction = predict_tomorrows_prices(
            model=model,
            market=market,
            district=district,
            variety=variety,
            grade=grade,
            current_data=data
        )
        time.sleep(10)

        return jsonify({
            "status": "success",
            "prediction": prediction
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9500)





