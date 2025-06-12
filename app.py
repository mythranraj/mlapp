from flask import Flask, request, render_template, redirect
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
DATA_FILE = 'data.csv'
model = LinearRegression()

# Load or initialize the dataset
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=['study_hours', 'previous_score', 'final_score'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    study_hours = float(request.form['study_hours'])
    previous_score = float(request.form['previous_score'])

    if df.shape[0] < 2:
        return "Not enough data to predict. Please upload a dataset and train the model."

    model.fit(df[['study_hours', 'previous_score']], df['final_score'])
    predicted = model.predict([[study_hours, previous_score]])[0]
    return f"<h2 style='color:white;background:#1f1f2e;padding:20px;'>📌 Predicted Final Score: {predicted:.2f}</h2><a href='/'>Back</a>"

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    new_data = pd.read_csv(file)
    global df
    df = new_data
    df.to_csv(DATA_FILE, index=False)
    return redirect('/')

@app.route('/train')
def train():
    if df.shape[0] < 2:
        return "Not enough data to train."
    model.fit(df[['study_hours', 'previous_score']], df['final_score'])
    return "✅ Model trained successfully! <a href='/'>Back</a>"

@app.route('/dataset')
def dataset():
    return df.to_html(classes='table', border=0) 

if __name__ == '__main__':
    app.run(debug=True)
