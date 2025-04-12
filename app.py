from flask import Flask, render_template, request # type: ignore
import pickle
import pandas as pd # type: ignore
import requests  # type: ignore # Import requests to send Telegram messages

app = Flask(__name__)

# Load the trained model and preprocessor
with open('best_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Define your Telegram boken and chat ID
TELEGRAM_BOT_TOKEN = '7872815366:AAE8sYt-smrw3-6BEfVlVd6fIPXPhWgPia8'  # Replace with your bot token
CHAT_ID = '918784405'  # Replace with your chat ID

# Function to send a message to Telegram
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    requests.post(url, data=payload)

# Define a function to preprocess input data
def preprocess_input(data):
    # Convert data to DataFrame
    input_data = pd.DataFrame(data, index=[0])
    
    # Ensure data types and column order
    input_data['daddr'] = input_data['daddr'].astype(str)
    input_data['saddr'] = input_data['saddr'].astype(str)
    input_data['category'] = input_data['category'].astype(str)
    input_data['subcategory'] = input_data['subcategory'].astype(str)
    
    # Apply preprocessor
    input_transformed = preprocessor.transform(input_data)
    
    return input_transformed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        new_data = {
            'pkSeqID': float(request.form['pkSeqID']),
            'proto': request.form['proto'],
            'saddr': request.form['saddr'],
            'sport': float(request.form['sport']),
            'daddr': request.form['daddr'],
            'dport': float(request.form['dport']),
            'seq': float(request.form['seq']),
            'stddev': float(request.form['stddev']),
            'N_IN_Conn_P_SrcIP': float(request.form['N_IN_Conn_P_SrcIP']),
            'min': float(request.form['min']),
            'state_number': float(request.form['state_number']),
            'mean': float(request.form['mean']),
            'N_IN_Conn_P_DstIP': float(request.form['N_IN_Conn_P_DstIP']),
            'drate': float(request.form['drate']),
            'srate': float(request.form['srate']),
            'max': float(request.form['max']),
            'category': request.form['category'],
            'subcategory': request.form['subcategory'],
        }

        try:
            # Preprocess input data
            input_transformed = preprocess_input(new_data)

            # Make predictions
            prediction = model.predict(input_transformed)

            # Send a Telegram message if prediction is 1
            if prediction[0] == 1:
                send_telegram_message("Alert: A potential victim has been detected!")

            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            error_message = "An error occurred: {}".format(str(e))
            return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
