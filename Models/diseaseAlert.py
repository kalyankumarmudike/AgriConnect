from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from geopy.distance import geodesic
from datetime import datetime
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')  # or your MongoDB URI
db = client.crop_disease
reports_collection = db.disease_reports
farmers_collection = db.farmers

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        image_file = request.files.get('image')
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))

        if not image_file:
            return jsonify({'error': 'No image provided'}), 400

        # Mock prediction
        prediction_result = {
            'disease': 'Leaf Blight',
            'severity': 'high',  # could be 'low', 'medium', 'high'
            'confidence': 0.92,
            'treatment': 'Apply fungicide XYZ'
        }

        # Store and alert if severity is medium or high
        if prediction_result['severity'] in ['medium', 'high']:
            store_disease_report(latitude, longitude, prediction_result)
            alert_nearby_farmers(latitude, longitude, prediction_result)

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def store_disease_report(latitude, longitude, prediction_result):
    report = {
        'latitude': latitude,
        'longitude': longitude,
        'disease_name': prediction_result['disease'],
        'severity': prediction_result['severity'],
        'confidence': prediction_result['confidence'],
        'timestamp': datetime.utcnow()
    }
    reports_collection.insert_one(report)

def alert_nearby_farmers(latitude, longitude, prediction_result):
    farmers = list(farmers_collection.find({}))
    nearby_farmers = []

    for farmer in farmers:
        dist = geodesic((latitude, longitude), (farmer['latitude'], farmer['longitude'])).kilometers
        if dist <= 2.0:
            farmer['distance'] = dist
            nearby_farmers.append(farmer)

    for farmer in nearby_farmers:
        if 'email' in farmer:
            send_email_alert(farmer, prediction_result, latitude, longitude)
        if 'phone' in farmer:
            send_sms_alert(farmer, prediction_result, latitude, longitude)

def send_email_alert(farmer, prediction_result, lat, lng):
    try:
        smtp_server = "smtp.example.com"
        smtp_port = 587
        smtp_username = "your_email@example.com"
        smtp_password = "your_password"

        message = MIMEMultipart()
        message['From'] = smtp_username
        message['To'] = farmer['email']
        message['Subject'] = f"URGENT: Crop Disease Alert in Your Area"

        body = f"""
        Dear {farmer['name']},

        A {prediction_result['severity']} severity case of {prediction_result['disease']} has been 
        detected approximately {farmer['distance']:.1f} km from your farm.

        Recommended treatment: {prediction_result['treatment']}

        Please inspect your crops as soon as possible.

        This is an automated alert from the Crop Disease Monitoring System.
        """

        message.attach(MIMEText(body, 'plain'))

        # Uncomment to actually send
        # server = smtplib.SMTP(smtp_server, smtp_port)
        # server.starttls()
        # server.login(smtp_username, smtp_password)
        # server.send_message(message)
        # server.quit()

        print(f"Email alert sent to {farmer['name']}")

    except Exception as e:
        print(f"Email error: {e}")

def send_sms_alert(farmer, prediction_result, lat, lng):
    # Placeholder: Use Twilio or similar
    print(f"SMS alert would be sent to {farmer['name']} at {farmer['phone']}")

@app.route('/heatmap-data', methods=['GET'])
def get_heatmap_data():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "latitude": "$latitude",
                        "longitude": "$longitude",
                        "disease_name": "$disease_name",
                        "severity": "$severity"
                    },
                    "count": {"$sum": 1}
                }
            }
        ]
        results = list(reports_collection.aggregate(pipeline))
        heatmap_data = []

        for r in results:
            severity = r['_id']['severity']
            count = r['count']
            weight = 1
            if severity == 'medium':
                weight = 5
            elif severity == 'high':
                weight = 10

            heatmap_data.append({
                'lat': r['_id']['latitude'],
                'lng': r['_id']['longitude'],
                'weight': weight * count,
                'disease': r['_id']['disease_name'],
                'severity': severity,
                'count': count
            })

        return jsonify(heatmap_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
