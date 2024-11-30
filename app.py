from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential
import librosa
import pickle
import os
import cv2
import base64
import numpy as np
from sentiment_analysis.sentiment_analysis import load_sentiment_model, analyze_sentiments, analyze_whatsapp_chat, plot_sentiment_results
import google.generativeai as genai
from speech_analysis.speech_analysis import speech_to_text, analyze_speech_emotion, convert_audio_to_wav
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Load Facial Emotion Recognition Model
json_file = open('facialrecognition/facialemotionmodel.json', 'r')
facial_model_json = json_file.read()
json_file.close()
facial_model = model_from_json(facial_model_json)
facial_model.load_weights('facialrecognition/facialemotionmodel.h5')

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Load Sentiment Analysis Model for WhatsApp Chat
sentiment_model = load_sentiment_model()

# Configure Google Generative AI
genai.configure(api_key="AIzaSyDDQunQPh9EqoIhizXvJD2jI3Dcf_YZtgA")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_facial_emotion', methods=['POST'])
def analyze_facial_emotion():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []
    
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    for (p, q, r, s) in faces:
        face_image = gray[q:q+s, p:p+r]
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        pred = facial_model.predict(img)
        prediction_label = labels[pred.argmax()]
        emotions.append(prediction_label)
    
    return jsonify({'emotions': emotions})

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if 'chat' not in request.form:
        return jsonify({'error': 'No chat text provided'})
    chat_text = request.form['chat']
    # Call the analyze_sentiments function from sentiment_analysis.py
    sentiment = analyze_sentiments(chat_text, sentiment_model)
    return jsonify({'sentiment': sentiment})

@app.route('/analyze_whatsapp_chat', methods=['POST'])
def analyze_whatsapp_chat_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the uploads directory exists
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Save the uploaded file temporarily
    file_path = os.path.join(uploads_dir, file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file'}), 500

    # Analyze WhatsApp chat using the sentiment_analysis module
    try:
        pos, neg, opinion = analyze_whatsapp_chat(file_path, sentiment_model)
        print(f"Analysis Result - Positive: {pos}, Negative: {neg}, Opinion: {opinion}")

        # Plot the sentiment analysis results and save to a BytesIO object
        buffer = BytesIO()
        plot_sentiment_results(pos, neg)
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()  # Close the plot to free memory

        # Encode image to base64 to send alongside the JSON response
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Send both sentiment data and image
        return jsonify({
            'positive': pos,
            'negative': neg,
            'opinion': opinion,
            'image': img_base64
        })

    except Exception as e:
        print(f"Error analyzing WhatsApp chat: {e}")
        return jsonify({'error': 'Error processing WhatsApp chat'}), 500



@app.route('/analyze_speech_emotion', methods=['POST'])
def analyze_speech_emotion_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Debugging: Log the file
    print(f"Received file: {file.filename}")

    # Ensure the uploads directory exists
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Save the uploaded file temporarily
    file_path = os.path.join(uploads_dir, file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file'}), 500

    # Convert audio to WAV format if needed
    try:
        wav_file_path = convert_audio_to_wav(file_path)
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        return jsonify({'error': 'Error converting audio to WAV'}), 500

    # Analyze speech emotion using the speech_analysis module
    try:
        sentiment = analyze_speech_emotion(wav_file_path, sentiment_model)
        print(f"Sentiment from analysis: {sentiment}")  # Log the sentiment
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        print(f"Error analyzing speech: {e}")
        return jsonify({'error': 'Error processing speech'}), 500



@app.route('/multimodal_analysis', methods=['POST'])
def multimodal_analysis():
    facial_emotion = request.form.get('facial_emotion')
    voice_emotion = request.form.get('voice_emotion')
    sentiment = request.form.get('sentiment')
    
    prompt = f"""Prompt:
    Input:
    
    Sentiment Analysis of WhatsApp Texts: {sentiment}
    Voice Emotion Detection: {voice_emotion}
    Facial Expression Analysis: {facial_emotion}
    Task:
    
    Based on the provided multimodal data, analyze the individual's emotional state. Assess the potential for mental health concerns such as depression or anxiety. Provide a comprehensive analysis and suggest appropriate actions."""
    
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return jsonify({'analysis': response.text})

if __name__ == '__main__':
    app.run(debug=True)