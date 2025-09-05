from src.chatbot import Chatbot
from src.model import Model  # Import the Model class
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for text processing
import pandas as pd  # Import pandas for data handling
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Initialize the chatbot instance
model = Model()  # Initialize the model
model.build_model()  # Build the model before using it
vectorizer = CountVectorizer()  # Initialize CountVectorizer
training_data = pd.read_csv('data/input.csv')
X_train = training_data['input_text']  # Features
y_train = training_data['response']  # Labels
X_train = vectorizer.fit_transform(X_train)  # Convert text data to numerical format
model.train_model(X_train, y_train)  # Train the model with the numerical data
chatbot = Chatbot(model, vectorizer)  # Pass the model and vectorizer to the Chatbot

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = str(chatbot.get_response(user_input))  # Ensure response is a string
    return jsonify({'response': response})

@app.route('/data', methods=['GET'])
def get_data():
    training_data = pd.read_csv('data/input.csv')
    return training_data.to_json(orient='records')

def main():
    # Start the Flask server
    app.run(debug=True)

if __name__ == "__main__":
    main()
