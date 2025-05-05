import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    """Train and save the tip prediction model using Seaborn's 'tips' dataset."""
    # Load the dataset
    tips = sns.load_dataset('tips')

    # Convert categorical features to numeric
    tips['day'] = tips['day'].astype('category').cat.codes
    tips['time'] = tips['time'].astype('category').cat.codes

    # Select features and target variable
    X = tips[['total_bill', 'size', 'day', 'time']]
    y = tips['tip']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully!")

def load_model():
    """Load the trained model from file."""
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        print("No trained model found. Training a new one...")
        train_model()
        return load_model()

# Run model training only if this script is executed directly
if __name__ == "__main__":
    train_model()