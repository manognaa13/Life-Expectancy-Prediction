from flask import Flask, render_template, request
import numpy as np
import pickle

with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
app = Flask(__name__)
app.template_folder = 'templates'  
app.static_url_path = '/static'  # The URL prefix for static files
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('first.html') 

@app.route('/index')
def first():
    return render_template('index.html');

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = {
        'Adult Mortality': float(data['Adult Mortality']),
        'Alcohol': float(data['Alcohol']),
        'Percentage Expenditure': float(data['Percentage Expenditure']),
        'Hepatitis B': float(data['Hepatitis B']),
        'Measles': float(data['Measles']),
        'BMI': float(data['BMI']),
        'Polio': float(data['Polio']),
        'Total expenditure': float(data['Total expenditure']),
        'Diphtheria': float(data['Diphtheria']),
        'HIV/AIDS': float(data['HIV/AIDS']),
        'GDP': float(data['GDP']),
        'Population': float(data['Population']),
        'Income composition of resources': float(data['Income composition of resources']),
        'Schooling': float(data['Schooling'])
    }
    
    input_data_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_data_array)[0]

    return render_template('last.html', prediction=prediction, user_inputs=input_data)

if __name__ == '__main__':
    app.run(debug=True)