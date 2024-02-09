import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load your training data
# Assuming your DataFrame is named 'train_df'
text_features = ['Top' + str(i) for i in range(1, 26)]
numerical_features = ['Stochastic_K', 'Stochastic_D', 'Momentum', 'Rate_of_Change', 'William_R', 'A/D_Oscillator', 'Disparity_5']

# Assuming 'Label' is your target variable
target_variable = 'Label'

# Combine features for the preprocessor
features = text_features + numerical_features

# Define the preprocessor class
class GPT2EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for row in X.values:
            row = [str(element) for element in row]
            combined_headlines = " ".join(row)
            inputs = self.tokenizer(combined_headlines, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            embeddings.append(embedding)

        return np.array(embeddings)

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', GPT2EmbeddingTransformer(), text_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Fit the preprocessor with your training data
X_train = train_df[features]
y_train = train_df[target_variable]
preprocessor.fit(X_train)

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')
