import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(data):
    # Update column names to match the CSV file
    data = data.rename(columns={'input_text': 'input', 'response': 'output'})
    
    # Remove missing values
    data = data.dropna()
    
    # Convert input and output to lowercase
    data['input'] = data['input'].str.lower()
    data['output'] = data['output'].str.lower()
    
    # Convert text data to numeric format
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['input'])
    
    # Ensure output labels do not contain NaN values
    y = data['output'].dropna().values
    
    return X, y  # Return features and labels
