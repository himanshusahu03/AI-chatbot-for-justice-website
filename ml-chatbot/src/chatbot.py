class Chatbot:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer  # Store vectorizer as an instance variable

    def get_response(self, user_input):
        # Preprocess the user input
        processed_input = self.preprocess_input(user_input)
        # Generate a response using the model
        processed_input = self.vectorizer.transform([processed_input])  # Transform input to numerical format
        response = self.model.predict(processed_input)
        return response

    def preprocess_input(self, user_input):
        # Implement preprocessing steps (e.g., tokenization, vectorization)
        # Example: Convert to lowercase
        processed_input = user_input.lower()
        return processed_input  # Updated to include basic preprocessing logic

    def load_model(self, model_path):
        # Load the trained model from the specified path
        import joblib
        self.model = joblib.load(model_path)  # Implemented model loading logic

    def save_model(self, model_path):
        # Save the trained model to the specified path
        pass  # Placeholder for model saving logic
