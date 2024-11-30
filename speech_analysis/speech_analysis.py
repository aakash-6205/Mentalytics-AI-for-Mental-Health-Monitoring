import sys
sys.path.append('D:/Major Project 7th SEM/EmotionAwareAI')  # Adjust the path accordingly
from pydub import AudioSegment
import speech_recognition as sr
import pickle
from sentiment_analysis.sentiment_analysis import clean

# Load Speech Sentiment Analysis Model
def load_sentiment_model():
    with open('sentiment_analysis/model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    return classifier

def convert_audio_to_wav(input_file):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(input_file)
    # Convert to WAV format
    output_file = "converted_audio.wav"
    audio.export(output_file, format="wav")
    return output_file

# Convert Speech to Text
def speech_to_text(audio_file):
    """
    Opens and listens to an audio file and translates it to text.
    Args:
        audio_file (file-like object): Uploaded audio file.
    Returns:
        str: Transcribed text from the audio file.
    """
     # Convert to WAV format if not already in correct format
    audio_file = convert_audio_to_wav(audio_file)  # Convert to WAV format
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            print("Loading audio file...")
            audio_text = r.record(source)  # records the entire audio file
            print('Converting audio transcripts into text...')
            text = r.recognize_google(audio_text)
            return text
    except Exception as e:
        print(f'Sorry, an error occurred: {e}')
        return None

# Analyze Speech Emotion Sentiments
def analyze_speech_emotion(audio_file, model):
    """
    Analyze the sentiment of a given audio file by converting speech to text.
    Args:
        audio_file (file-like object): Uploaded audio file.
        model: Pre-trained sentiment analysis model.
    Returns:
        str: Sentiment classification ('positive' or 'negative').
    """
    # Convert speech to text
    transcribed_text = speech_to_text(audio_file)
    if not transcribed_text:
        return 'Error: Speech to text conversion failed'
    
    # Perform sentiment analysis on the transcribed text
    features = clean(transcribed_text.split())
    sentiment = model.classify(features)
    return sentiment

if __name__ == "__main__":
    model = load_sentiment_model()
    
    # Example usage:
    audio_path = 'speech_analysis/Roshan-pos.wav'
    sentiment = analyze_speech_emotion(audio_path, model)
    print(f"Sentiment: {sentiment}")