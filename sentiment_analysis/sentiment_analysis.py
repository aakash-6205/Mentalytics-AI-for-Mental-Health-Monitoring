import pickle
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import nltk.classify.util
import matplotlib.pyplot as plt
import numpy as np

def clean(words):
    return dict([(word, True) for word in words])

# Load the Sentiment Analysis Model
def load_sentiment_model():
    with open('sentiment_analysis/model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    return classifier

# Analyze Sentiments of a Given Text
def analyze_sentiments(chat_text, model):
    features = clean(chat_text.split())
    sentiment = model.classify(features)
    return sentiment

# Analyze Sentiments of a WhatsApp Chat File
def analyze_whatsapp_chat(file_path, model):
    opinion = {}
    pos, neg = 0, 0

    with open(file_path, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding here
        for line in f:
            try:
                chat = line.split('-')[1].split(':')[1]
                name = line.split('-')[1].split(':')[0]
                if opinion.get(name, None) is None:
                    opinion[name] = [0, 0]
                res = model.classify(clean(chat.split()))
                if res == 'positive':
                    pos += 1
                    opinion[name][0] += 1
                else:
                    neg += 1
                    opinion[name][1] += 1
            except IndexError:
                pass

    return pos, neg, opinion


# Plot Sentiment Analysis Results
def plot_sentiment_results(pos, neg):
    neg = abs(neg)
    labels = ['positive', 'negative']
    sizes = [pos, neg]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('WhatsApp Sentiment Analysis')
    plt.show()

if __name__ == "__main__":
    model = load_sentiment_model()
    
    # Example usage:
    chat_text = "i don't love you"
    sentiment = analyze_sentiments(chat_text, model)
    print(f"Sentiment: {sentiment}")
    
    # Analyze a WhatsApp Chat File
    pos, neg, opinion = analyze_whatsapp_chat('sentiment_analysis/WhatsApp Chat with Aakash Kumar Kiit.txt', model)
    print(f"Positive: {pos}, Negative: {neg}")
    plot_sentiment_results(pos, neg)