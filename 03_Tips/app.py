from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
def load_and_train_model():
    tips = sns.load_dataset('tips')

    # Convert categorical features to numeric
    tips['day'] = tips['day'].astype('category').cat.codes
    tips['time'] = tips['time'].astype('category').cat.codes

    # Select the features and target variable
    X = tips[['total_bill', 'size', 'day', 'time']]
    y = tips['tip']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Ensure model exists, otherwise train and save it
if not os.path.exists('model.pkl'):
    load_and_train_model()

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs
        total_bill = float(request.form.get('total_bill', 0))
        size = int(request.form.get('size', 1))
        day = request.form.get('day', 'Fri')  # Default day
        time = request.form.get('time', 'Dinner')  # Default time

        # Convert categorical inputs to numeric
        day_mapping = {'Thur': 0, 'Fri': 1, 'Sat': 2, 'Sun': 3}
        time_mapping = {'Lunch': 0, 'Dinner': 1}

        day_numeric = day_mapping.get(day, 1)  # Default to Friday if unrecognized
        time_numeric = time_mapping.get(time, 1)  # Default to Dinner if unrecognized

        # Make a prediction using the model
        prediction = model.predict([[total_bill, size, day_numeric, time_numeric]])[0]

        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)


