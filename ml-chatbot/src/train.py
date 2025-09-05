import pandas as pd
from data_preprocessing import load_data, preprocess_data
from model import Model
from utils import save_model

def main():
    # Load and preprocess the data
    data = load_data('data/input.csv')
    X, y = preprocess_data(data)

    # Initialize and train the model
    model = Model()
    model.build_model()
    model.train_model(X, y)

    # Save the trained model
    save_model(model, 'model/chatbot_model.h5')

if __name__ == "__main__":
    main()
