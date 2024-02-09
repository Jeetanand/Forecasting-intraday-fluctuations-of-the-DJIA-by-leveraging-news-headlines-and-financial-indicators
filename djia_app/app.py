import streamlit as st
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from transformers import GPT2Tokenizer, GPT2Model
import torch
import yfinance as yf

# Load the saved LSTM model
model = load_model('lstm_model_advance.h5')

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

# Define the GPT2EmbeddingTransformer
class GPT2EmbeddingTransformer:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def transform(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embedding

# Function to calculate additional features
def calculate_additional_features(df):
    # Calculate technical indicators using pandas_ta
    df.ta.stoch(append=True)
    df.ta.mom(append=True)
    df.ta.roc(append=True)
    df.ta.willr(append=True)
    df.ta.AD(append=True)
    df.ta.wma(append=True, length=5)

    # Return the selected features
    return df[['Stoch_%k', 'Stoch_%d', 'Mom', 'ROC', 'Willr', 'AD', 'WMA_5']]

# App layout
st.title('DJIA Prediction App')
st.sidebar.header('User Input')

# Input for news headlines
user_input_text = st.sidebar.text_area('Enter 25 headlines (one per line):')

# Input for date
user_input_date = st.sidebar.date_input('Select a date:', pd.to_datetime('today'))

# Process user input
if user_input_text and user_input_date:
    try:
        # Fetch DJIA historical data for the selected date
        djia_data = yf.download("^DJI", start=user_input_date, end=user_input_date)

        if djia_data.empty:
            st.warning('No data available for the selected date. Please choose another date.')
        else:
            # Calculate additional features
            additional_features = calculate_additional_features(djia_data)

            # Extract headlines
            headlines = user_input_text.strip().split('\n')

            # Create a DataFrame with the user input
            user_data = pd.DataFrame({'Top' + str(i): headlines[i - 1] for i in range(1, 26)}, index=[0])
            user_data['Date'] = user_input_date.strftime('%Y-%m-%d')

            # Merge user data with additional features
            user_data = pd.concat([user_data, additional_features], axis=1)

            # Define the features and the target variable
            text_features = ['Top' + str(i) for i in range(1, 26)]
            numerical_features = ['Stoch_%k', 'Stoch_%d', 'Mom', 'ROC', 'Willr', 'AD', 'WMA_5']

            # Define the preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('text', GPT2EmbeddingTransformer(tokenizer, gpt2_model), text_features),
                    ('num', StandardScaler(), numerical_features)
                ]
            )

            # Apply the preprocessing pipeline
            user_transformed = preprocessor.transform(user_data)
            user_transformed = user_transformed.reshape((user_transformed.shape[0], user_transformed.shape[1], 1))

            # Make predictions using the loaded LSTM model
            prediction = (model.predict(user_transformed) > 0.5).astype(int)[0, 0]

            # Display prediction result
            st.subheader('Prediction Result:')
            st.write(f'The predicted DJIA movement is: {"Up" if prediction == 1 else "Down"}')

    except ValueError:
        st.warning('Invalid input or no data available. Please check your input and try again.')
else:
    st.info('Please provide both news headlines and a date to make predictions.')
