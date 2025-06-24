from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import requests
import datetime
import statistics
from collections import Counter
import os  
from flask_cors import CORS
from g4f.client import Client
import json

WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', 'JNFME4DMT4KBURA9YSUF7GA2R') 
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDIgdcjFOMSHu7D3j7MH23v4ZkRaUWC0l8') 

client = Client()

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app, resources={r"/cropRotation": {"origins": "*"}, r"/sample": {"origins": "*"}})

def get_agricultural_weather_forecast(latitude, longitude, days=120):
    """
    Get agricultural weather forecast for a specific location

    Args:
        latitude: Location latitude
        longitude: Location longitude
        days: Number of days to forecast (default: 120)

    Returns:
        Weather summary and daily forecast data
    """

    start_date = datetime.date.today()
    end_date = start_date + datetime.timedelta(days=days)

    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}/{start_date}/{end_date}?key={WEATHER_API_KEY}&unitGroup=metric&include=days"

    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        
        daily_forecasts = []
        temperatures = []
        precipitation = []
        humidity = []
        wind_speeds = []
        conditions = []

        for day in data['days']:
            date = day['datetime']
            temp = day['temp']
            max_temp = day['tempmax']
            min_temp = day['tempmin']
            desc = day['conditions']
            precip = day.get('precip', 0)
            humid = day.get('humidity', 0)
            wind = day.get('windspeed', 0)

            daily_forecasts.append({
                'date': date,
                'conditions': desc,
                'temp': temp,
                'tempmax': max_temp,
                'tempmin': min_temp,
                'precip': precip,
                'humidity': humid,
                'windspeed': wind
            })

            temperatures.append(temp)
            precipitation.append(precip)
            humidity.append(humid)
            wind_speeds.append(wind)
            conditions.append(desc)

        
        summary = {
            'location': f"{latitude}, {longitude}",
            'address': data.get('resolvedAddress', 'Unknown'),
            'period': f"{start_date} to {end_date}",
            'total_days': len(daily_forecasts),
            'avg_temp': round(statistics.mean(temperatures), 1),
            'max_temp': max(day['tempmax'] for day in data['days']),
            'min_temp': min(day['tempmin'] for day in data['days']),
            'total_precip': round(sum(precipitation), 1),
            'avg_humidity': round(statistics.mean(humidity), 1),
            'avg_wind': round(statistics.mean(wind_speeds), 1),
            'common_conditions': Counter(conditions).most_common(3),
            'monthly_breakdown': get_monthly_breakdown(daily_forecasts)
        }

        return summary, daily_forecasts
    else:
        print("Error:", response.status_code, response.text)
        return None, None

def get_monthly_breakdown(daily_forecasts):
    """Generate monthly summaries of the forecast data"""
    months = {}

    for day in daily_forecasts:
        date_obj = datetime.datetime.strptime(day['date'], '%Y-%m-%d')
        month_key = date_obj.strftime('%Y-%m')
        month_name = date_obj.strftime('%B %Y')

        if month_key not in months:
            months[month_key] = {
                'name': month_name,
                'temps': [],
                'precip': [],
                'humidity': [],
                'days': 0
            }

        months[month_key]['temps'].append(day['temp'])
        months[month_key]['precip'].append(day['precip'])
        months[month_key]['humidity'].append(day['humidity'])
        months[month_key]['days'] += 1

    
    monthly_summary = []
    for key, month in months.items():
        monthly_summary.append({
            'month': month['name'],
            'avg_temp': round(statistics.mean(month['temps']), 1),
            'total_precip': round(sum(month['precip']), 1),
            'avg_humidity': round(statistics.mean(month['humidity']), 1),
            'days': month['days']
        })

    return monthly_summary

def get_agricultural_insights(summary):
    """Generate agricultural insights based on weather forecast"""
    insights = []

    # Temperature insights
    if summary['max_temp'] > 35:
        insights.append("⚠️ High temperatures above 35°C may cause heat stress in crops. Consider shade or additional irrigation.")

    if summary['min_temp'] < 5:
        insights.append("⚠️ Low temperatures below 5°C may affect cold-sensitive crops. Monitor frost forecasts.")

    
    monthly_precip = [month['total_precip'] for month in summary['monthly_breakdown']]
    if summary['total_precip'] < 100:
        insights.append("⚠️ Low total precipitation expected. Consider irrigation planning.")
    elif summary['total_precip'] > 500:
        insights.append("⚠️ High precipitation expected. Monitor for potential flooding or erosion.")

    if max(monthly_precip) - min(monthly_precip) > 100:
        insights.append("📊 Significant variation in monthly rainfall. Plan for both dry and wet periods.")

    
    if summary['avg_humidity'] > 80:
        insights.append("⚠️ High average humidity may increase disease pressure. Monitor crops for fungal diseases.")
    elif summary['avg_humidity'] < 40:
        insights.append("⚠️ Low humidity may increase water requirements. Monitor soil moisture closely.")

    
    if summary['avg_wind'] > 20:
        insights.append("💨 Higher than average winds may increase evaporation. Consider wind breaks or increased irrigation.")

    return insights

def format_weather_summary(summary):
    """Format weather summary into a string for Gemini prompt"""
    output = []
    output.append("===== AGRICULTURAL WEATHER FORECAST =====")
    output.append(f"Location: {summary['address']} ({summary['location']})")
    output.append(f"Period: {summary['period']} ({summary['total_days']} days)")

    output.append("\n----- OVERALL SUMMARY -----")
    output.append(f"Average Temperature: {summary['avg_temp']}°C")
    output.append(f"Maximum Temperature: {summary['max_temp']}°C")
    output.append(f"Minimum Temperature: {summary['min_temp']}°C")
    output.append(f"Total Precipitation: {summary['total_precip']}mm")
    output.append(f"Average Humidity: {summary['avg_humidity']}%")
    output.append(f"Average Wind Speed: {summary['avg_wind']} km/h")

    output.append("\nCommon Weather Conditions:")
    for condition, count in summary['common_conditions']:
        percentage = (count / summary['total_days']) * 100
        output.append(f"  - {condition}: {count} days ({percentage:.1f}%)")

    output.append("\n----- MONTHLY BREAKDOWN -----")
    for month in summary['monthly_breakdown']:
        output.append(f"\n{month['month']} ({month['days']} days):")
        output.append(f"  Avg Temp: {month['avg_temp']}°C")
        output.append(f"  Total Precipitation: {month['total_precip']}mm")
        output.append(f"  Avg Humidity: {month['avg_humidity']}%")

    return "\n".join(output)

def format_soil_info(soil_type, npk_values):
    """Format soil information for Gemini prompt"""
    output = []
    output.append("\n===== SOIL INFORMATION =====")
    output.append(f"Soil Type: {soil_type}")
    output.append("NPK Values:")
    output.append(f"  - Nitrogen (N): {npk_values['N']}%")
    output.append(f"  - Phosphorus (P): {npk_values['P']}%")
    output.append(f"  - Potassium (K): {npk_values['K']}%")
    output.append(f"  - pH: {npk_values.get('pH', 'Not measured')}")
    output.append(f"  - Organic Matter: {npk_values.get('organic_matter', 'Not measured')}%")

    return "\n".join(output)

# def get_crop_recommendations(location, weather_summary, soil_info, previous_crop=None):
#     """Generate crop recommendations using Gemini API"""

#     prompt = f"""
# As an agricultural expert, I need your recommendations for crop planning based on the following data:

# {weather_summary}

# {soil_info}

# {'Previous Crop: ' + previous_crop if previous_crop else 'No previous crop information available.'}

# Based on this information for the location: {location}, please provide:
# 1. Top 5 recommended crops for this soil type and weather forecast
# 2. For each crop, explain why it's suitable based on the weather and soil conditions
# 3. Planting schedule recommendations (when to plant each crop in a general timeframe)
# 4. Any specific warnings or considerations for these crops
# 5. Recommended soil amendments or fertilizer adjustments based on the NPK values
# 6. Irrigation recommendations based on the precipitation forecast

# Please format your response in a structured, easy-to-read format. 
# """

    
#     model = genai.GenerativeModel('gemini-2.0-flash')
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         print(f"Error calling Gemini API: {e}")
#         return "Failed to get crop recommendations."










from g4f.client import Client
import json

client = Client()

def get_crop_recommendations(location, weather_summary, soil_info, previous_crop=None):
    """Generate structured crop recommendations using GPT-4 API via g4f and return as clean JSON"""
    
    prompt = f"""
As an agricultural expert, analyze the following data and respond strictly in the JSON format provided below — no additional commentary or explanation outside the JSON:

Location: {location}

Weather Summary:
{weather_summary}

Soil Information:
{soil_info}

{'Previous Crop: ' + previous_crop if previous_crop else 'No previous crop information available.'}

Respond in this strict JSON structure only:

{{
  "summary": {{
    "top_5_recommended_crops": [],
    "suitability_justification": {{"crop_name": "Explain for each crop why it's suitable"}},
    "planting_schedule_recommendations": {{"crop_name": "Timeframe to plant each crop"}},
    "warnings_and_considerations": {{"crop_name": "Any pests, diseases, or regional issues for each crop"}},
    "Recommended Soil Amendments and Fertilizer Adjustments": {{"crop_name": "Based on NPK values for each crop"}},
    "Irrigation_Recommendations": {{"crop_name": "Based on precipitation and needs for each crop"}},
    "Methods": {{"crop_name": "Modern methods or traditional practices to follow for each crop"}},
    "Additional Considerations": []
  }}
}}

Important:
- Return ONLY valid JSON parsable with json.loads()
- Use double quotes for all JSON keys and string values
- No markdown formatting or code blocks
- No additional explanations outside the JSON structure
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the JSON string (unescaping \n and \")
        raw_text = response.choices[0].message.content
        parsed_json = json.loads(raw_text)

        return parsed_json

    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return {"error": "Failed to get crop recommendations."}










def get_crop_recommendations_gemini(location, weather_summary, soil_info, previous_crop=None):
    """Generate structured crop recommendations using Gemini API and return as clean JSON"""
    
    prompt = f"""
As an agricultural expert, analyze the following data and respond strictly in the JSON format provided below — no additional commentary or explanation outside the JSON:

Location: {location}

Weather Summary:
{weather_summary}

Soil Information:
{soil_info}

{'Previous Crop: ' + previous_crop if previous_crop else 'No previous crop information available.'}

Respond in this strict JSON structure only:

{{
  "summary": {{
    "top_5_recommended_crops": [],
    "suitability_justification": {{"crop_name": "Explain for each crop why it's suitable"}},
    "planting_schedule_recommendations": {{"crop_name": "Timeframe to plant each crop"}},
    "warnings_and_considerations": {{"crop_name": "Any pests, diseases, or regional issues for each crop"}},
    "Recommended Soil Amendments and Fertilizer Adjustments": {{"crop_name": "Based on NPK values for each crop"}},
    "Irrigation_Recommendations": {{"crop_name": "Based on precipitation and needs for each crop"}},
    "Methods": {{"crop_name": "Modern methods or traditional practices to follow for each crop"}},
    "Additional Considerations": []
  }}
}}

Important:
- Return ONLY valid JSON parsable with json.loads()
- Use double quotes for all JSON keys and string values
- No markdown formatting or code blocks
- No additional explanations outside the JSON structure
"""
    
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Extract the text response
        raw_text = response.text
        
        # Clean the response (Gemini sometimes adds markdown formatting)
        clean_text = raw_text.strip().replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        parsed_json = json.loads(clean_text)
        
        return parsed_json

    except json.JSONDecodeError as je:
        print(f"JSON parsing error: {je}")
        print(f"Raw response was: {raw_text}")
        return {"error": "Failed to parse the response as JSON."}
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": "Failed to get crop recommendations."}








@app.route('/cropRotation', methods=['POST'])
def index():
    """
    Main route for the API. Handles JSON input and returns results.
    """
    try:
        
        data = request.get_json()

        
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        soil_type = data['soil_type']
        npk_values = {
            "N": float(data['nitrogen']),
            "P": float(data['phosphorus']),
            "K": float(data['potassium']),
            "pH": float(data['ph']),
            "organic_matter": float(data['organic_matter'])
        }
        previous_crop = data['previous_crop']

        
        weather_summary, _ = get_agricultural_weather_forecast(latitude, longitude)

        if weather_summary:
            
            weather_text = format_weather_summary(weather_summary)
            soil_text = format_soil_info(soil_type, npk_values)

            
            # recommendations = get_crop_recommendations(
            #     weather_summary['address'],
            #     weather_text,
            #     soil_text,
            #     previous_crop
            # )
            recommendations = get_crop_recommendations_gemini(
                weather_summary['address'],
                weather_text,
                soil_text,
                previous_crop
            )
            insights = get_agricultural_insights(weather_summary)

            return jsonify({
                'recommendations': recommendations,
                'weather_summary': weather_summary,
                'soil_type': soil_type,
                'npk_values': npk_values,
                'insights': insights
            }), 200
        else:
            return jsonify({
                'error': 'Failed to get weather forecast. Please check your coordinates and try again.'
            }), 400

    except KeyError as e:
        return jsonify({'error': f'Missing key in JSON: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {e}. Ensure numeric values for latitude, longitude, and NPK.'}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': f'Unexpected error: {e}'}), 500



def get_soil_types():
    """Helper function to get soil types. Useful for populating dropdown."""
    return [
        "Sandy", "Clay", "Loamy", "Silt", "Sandy Loam", "Clay Loam",
        "Silty Clay", "Silty Loam", "Peaty", "Chalky", "Laterite", "Red Soil", "Black Soil"
    ]








@app.route('/sample', methods=['POST'])
def get_crop_recommendation():
    data = {
            "insights": [
                "⚠️ High temperatures above 35°C may cause heat stress in crops. Consider shade or additional irrigation.",
                "⚠️ High precipitation expected. Monitor for potential flooding or erosion.",
                "📊 Significant variation in monthly rainfall. Plan for both dry and wet periods.",
                "💨 Higher than average winds may increase evaporation. Consider wind breaks or increased irrigation."
            ],
            "npk_values": {
                "K": 75.0,
                "N": 50.0,
                "P": 25.0,
                "organic_matter": 2.0,
                "pH": 6.5
            },
            "recommendations": {
                "summary": {
                    "Additional Considerations": [],
                    "Irrigation_Recommendations": {
                        "Cotton": "Cotton requires minimal irrigation during the monsoon; however, ensure adequate water during the dry spells.",
                        "Groundnut": "Groundnuts need well-drained soil, but irrigation should be limited during the wettest months (July to August).",
                        "Maize": "Require moderate irrigation during dry periods, especially in May and September when precipitation is lower.",
                        "Pulses": "Pulses need light irrigation during the early stages; reduce irrigation during the rainy months (July to August).",
                        "Soybean": "Soybean requires sufficient moisture, but over-irrigation should be avoided during periods of heavy rainfall (June to August).",
                        "heading": "Based on precipitation and needs"
                    },
                    "Methods": {
                        "Cotton": "Implement integrated pest management (IPM) and use genetically resistant varieties to control bollworm.",
                        "Groundnut": "Adopt no-till farming practices to preserve soil moisture and reduce soil erosion.",
                        "Maize": "Utilize drip irrigation and precision farming techniques to optimize water usage.",
                        "Pulses": "Use minimal tillage and intercropping to improve soil health and reduce pest pressure.",
                        "Soybean": "Incorporate crop rotation and use integrated pest management to control pests naturally.",
                        "heading": "Modern methods or traditional practices to follow"
                    },
                    "Recommended Soil Amendments and Fertilizer Adjustments": {
                        "Cotton": "Cotton requires balanced fertilization, focusing on phosphorus (P) and potassium (K) to promote fiber development.",
                        "Groundnut": "Due to the high potassium (K) content, consider adding a nitrogen supplement and organic matter to promote root growth.",
                        "Maize": "Apply a balanced fertilizer with higher nitrogen content (N) to support growth during the vegetative stage.",
                        "Pulses": "For pulses, boost phosphorus (P) for stronger root systems and nitrogen (N) for vegetative growth.",
                        "Soybean": "Soybean benefits from phosphorus (P) for root development; a moderate dose of NPK can support overall growth.",
                        "heading": "Based on NPK values"
                    },
                    "planting_schedule_recommendations": {
                        "Cotton": "Plant in May to June, ensuring early growth before the heaviest rainfall periods begin.",
                        "Groundnut": "Plant in late May to early June, as groundnuts require warm soil for germination.",
                        "Maize": "Plant in late May to early June for optimal growth during the warmest months.",
                        "Pulses": "Best planted in June to early July, taking advantage of moderate rainfall and temperatures.",
                        "Soybean": "Ideal planting time is early June to late July, coinciding with the onset of monsoon rains.",
                        "heading": "Timeframe to plant each crop"
                    },
                    "suitability_justification": {
                        "Cotton": "Cotton requires a warm climate with a longer dry period, and its growth is supported by the adequate temperature range and occasional rains.",
                        "Groundnut": "Groundnut requires warm temperatures and well-drained soil, making it compatible with the loamy soil and favorable climatic conditions.",
                        "Maize": "Maize thrives in warmer climates with adequate moisture, especially given the summer temperatures and moderate precipitation.",
                        "Pulses": "Pulses perform well in varied climates and can tolerate periods of moderate drought, aligning with the seasonal precipitation fluctuations.",
                        "Soybean": "Soybean is suited for loamy soil with moderate moisture, and its growth benefits from the temperature and precipitation patterns during the season.",
                        "heading": "Explain for each crop why it's suitable"
                    },
                    "top_5_recommended_crops": [
                        "Maize",
                        "Soybean",
                        "Groundnut",
                        "Pulses",
                        "Cotton"
                    ],
                    "warnings_and_considerations": {
                        "Cotton": "Cotton crops are vulnerable to bollworm infestations and the spread of fungal diseases like wilt.",
                        "Groundnut": "Monitor for root rot and leaf spot diseases, especially during humid conditions.",
                        "Maize": "Watch for pests like stem borers and armyworms, as well as fungal diseases.",
                        "Pulses": "Be aware of aphid and whitefly infestations, which can impact the health of pulses.",
                        "Soybean": "Be cautious of rust and aphid infestations, which may affect crop yield.",
                        "heading": "Any pests, diseases, or regional issues"
                    }
                }
            },
            "soil_type": "Loamy Soil",
            "weather_summary": {
                "address": "17.385,78.4867",
                "avg_humidity": 64.6,
                "avg_temp": 28.3,
                "avg_wind": 23.7,
                "common_conditions": [
                    [
                        "Partially cloudy",
                        119
                    ],
                    [
                        "Rain, Partially cloudy",
                        2
                    ]
                ],
                "location": "17.385, 78.4867",
                "max_temp": 40.4,
                "min_temp": 22.6,
                "monthly_breakdown": [
                    {
                        "avg_humidity": 39.5,
                        "avg_temp": 32.9,
                        "days": 24,
                        "month": "May 2025",
                        "total_precip": 34.9
                    },
                    {
                        "avg_humidity": 61.8,
                        "avg_temp": 29.1,
                        "days": 30,
                        "month": "June 2025",
                        "total_precip": 146.1
                    },
                    {
                        "avg_humidity": 72.4,
                        "avg_temp": 26.7,
                        "days": 31,
                        "month": "July 2025",
                        "total_precip": 180.0
                    },
                    {
                        "avg_humidity": 77.1,
                        "avg_temp": 25.9,
                        "days": 31,
                        "month": "August 2025",
                        "total_precip": 217.2
                    },
                    {
                        "avg_humidity": 77.2,
                        "avg_temp": 26.1,
                        "days": 5,
                        "month": "September 2025",
                        "total_precip": 25.6
                    }
                ],
                "period": "2025-05-08 to 2025-09-05",
                "total_days": 121,
                "total_precip": 603.8
            }
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)