# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import torch
# from flask import Flask, request, jsonify
# import os
# os.makedirs('./temp', exist_ok=True)

# # Load the image processor and model
# processor = AutoImageProcessor.from_pretrained("Diginsa/Plant-Disease-Detection-Project", use_fast=True)
# model = AutoModelForImageClassification.from_pretrained("Diginsa/Plant-Disease-Detection-Project")


# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()
#     return model.config.id2label[predicted_class_idx]


# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     try:
#         file_path = f"./temp/{file.filename}"
#         file.save(file_path)
#         # Make prediction
#         prediction = predict(file_path)
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from google import genai
import PIL.Image
import os
import json 
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

client = genai.Client(api_key="AIzaSyDIgdcjFOMSHu7D3j7MH23v4ZkRaUWC0l8")  

def predict(image_path):
    image = PIL.Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=[
            """You are a plant disease detection expert.
            Given the image of a plant, analyze the visible symptoms and return a JSON object with the following structure and information:

            {
                "name": "",
                "crop": "",
                "description": "",
                "symptoms": [],
                "treatment": [],
                "preventionTips": [],
                "severity": "",
                "confidence":""
            }

            Respond only in valid JSON format. Do not include any other explanation or markdown ticks.
            """
            , image
        ]

    )
    return response.text

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = f"./temp/{file.filename}"
        os.makedirs('./temp', exist_ok=True)
        file.save(file_path)

        prediction_text = predict(file_path)
        print(f"Raw LLM Output: {prediction_text}") 
        prediction_text = prediction_text.strip().replace('```json', '').replace('```', '')
        try:
            prediction_json = json.loads(prediction_text) 
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}, Input: {prediction_text}")
            return jsonify({'error': 'Failed to decode JSON from model output', 'raw_output': prediction_text}), 500

        return jsonify({'prediction': prediction_json})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path) 



# @app.route('/sample_plant_disease', methods=['GET'])
# def get_plant_disease():
#     disease_data = {
        
#         "prediction": {
#             "crop": "Tomato",
#             "description": "Early blight is a common fungal disease that affects tomatoes and other plants in the Solanaceae family. It is caused by the fungus Alternaria solani and Alternaria tomatophila.",
#             "name": "Early Blight",
#             "preventionTips": [
#                 "Use disease-free seeds and transplants.",
#                 "Practice crop rotation to avoid build-up of the pathogen in the soil.",
#                 "Avoid overhead watering to reduce leaf wetness and humidity.",
#                 "Mulch around plants to prevent soil splash and reduce spread of the fungus.",
#                 "Monitor plants regularly for early signs of disease.",
#                 "Maintain proper soil fertility and drainage."
#             ],
#             "severity": "Moderate to High",
#             "symptoms": [
#                 "Dark brown to black circular spots on leaves, often with concentric rings (target-like appearance)",
#                 "Yellowing of the leaf tissue around the spots",
#                 "Spots may enlarge and merge, leading to leaf blight and defoliation",
#                 "Stem lesions can occur, particularly near the soil line",
#                 "Fruit can develop dark, sunken lesions near the stem end"
#             ],
#             "treatment": [
#                 "Remove and destroy infected leaves and plant debris to reduce inoculum.",
#                 "Apply a fungicide containing chlorothalonil, mancozeb, copper-based fungicides, or strobilurins (e.g., azoxystrobin) according to label instructions. Repeat applications may be necessary.",
#                 "Improve air circulation around plants by pruning and spacing appropriately.",
#                 "Ensure plants receive adequate nutrition to improve overall health and resistance."
#             ]
#         }
#     }
#     return jsonify(disease_data)

if __name__ == '__main__':
    app.run(debug=True)
