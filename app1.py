# ---- Import Libraries ----
from flask import Flask, render_template, request # type: ignore
import pickle
import pandas as pd # type: ignore
import requests # type: ignore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---- Initialize Flask App ----
app = Flask(__name__)

# ---- Load Trained Model and Preprocessor ----
with open('best_knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# ---- Telegram Bot Setup ----
TELEGRAM_BOT_TOKEN = '8028448864:AAGcqkSO-cgQw4tCtgUFNMpC5pvDBwkUUPg'
CHAT_ID = '1260757241'

# ---- Email Setup ----
EMAIL_ADDRESS = 'nagasailika88@gmail.com'  # Your Gmail
EMAIL_PASSWORD = 'rwua leda fbcu zpid'    # Gmail App Password (NOT your Gmail password)
TO_EMAIL = 'nagasailika88@gmail.com'   # Where you want to send alerts

# ---- Function to Send Telegram Message ----
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        requests.post(url, data=payload)
        print('Telegram alert sent successfully!')
    except Exception as e:
        print(f'Failed to send Telegram alert: {e}')

# ---- Function to Send Email Alert ----
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_alert(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print('Email alert sent successfully!')
    except Exception as e:
        print(f'Failed to send email alert: {e}')


# ---- Preprocess Input Data ----
def preprocess_input(data):
    input_data = pd.DataFrame(data, index=[0])

    # Ensure proper data types
    input_data['daddr'] = input_data['daddr'].astype(str)
    input_data['saddr'] = input_data['saddr'].astype(str)
    input_data['category'] = input_data['category'].astype(str)
    input_data['subcategory'] = input_data['subcategory'].astype(str)

    # Apply preprocessor
    input_transformed = preprocessor.transform(input_data)

    return input_transformed

# ---- Home Route ----
@app.route('/')
def home():
    return render_template('index.html')

# ---- Predict Route ----
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
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
            # Preprocess input
            input_transformed = preprocess_input(new_data)

            # Make prediction
            prediction = model.predict(input_transformed)

            # If attack is detected
            if prediction[0] == 1:
                send_telegram_message("ALERT: Potential Botnet Victim Detected!")
                send_email_alert(
                    subject="Botnet Attack Detected!",
                    body="A potential victim has been detected by your Botnet Detection System. Please take immediate action!"
                )

            # Show result
            return render_template('result.html', prediction=prediction[0])

        except Exception as e:
            error_message = "An error occurred: {}".format(str(e))
            return render_template('error.html', error_message=error_message)

# ---- Run the App ----
if __name__ == '__main__':
    app.run(debug=True)
