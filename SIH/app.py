from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import combined_script_no_comments as code
import joblib
import xgboost as xgb
import pandas as pd
import tensorflow as tf
import numpy as np
from itertools import groupby
# import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import h5py

import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical  # Replacing np_utils
import pandas as pd

import numpy as np
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import keras.src.preprocessing

app = Flask(__name__)
app.secret_key = "my_secret_key_1"


def reformat_string(hex_string):
    return ' '.join(hex_string[i:i+2] for i in range(0, len(hex_string), 2))


def predictor(ciphertext, tokenizer_path, model_path):
    formatted_array = np.array([reformat_string(ciphertext)])

    tokenizer=joblib.load(tokenizer_path)
    k=tokenizer.texts_to_sequences(formatted_array)
    maxlen=100
    padded_seq=pad_sequences(k,maxlen=maxlen,padding='post')
    model = load_model(model_path)
    return model.predict(padded_seq)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and (file.filename.endswith('.txt') or file.filename.endswith('.csv')):
                # Read the file content and process it
                uploaded_text = file.read().decode('utf-8').strip()
                 # Remove any \r\n characters from the text, as well as leading/trailing spaces
                uploaded_text = uploaded_text.replace('\r\n', '').replace('\n', '').replace('\r', '').strip()
                
                processed_text = uploaded_text.lower().replace(' ', '')

                # print(f"Processed text from file: {processed_text}")
                binary_string = code.transformer_str(processed_text)
                features = code.append_features_to_dataset(binary_string=binary_string)
                # print(f"Features extracted from file: {features}")
                session['features']=features
                
                return redirect(url_for('result', ciphertext=processed_text))


        # Get the text input from the form
        uploaded_text = request.form.get('text')
        if uploaded_text:
            # Process the text: Convert to lowercase and remove spaces
            processed_text = uploaded_text.lower().replace(' ', '')
            # print(processed_text)
            binary_string = code.transformer_str(processed_text)
            features = code.append_features_to_dataset(binary_string=binary_string)
            # print(features)
            session['features'] = features
            # session['ciphertext'] = processed_text

            # selected_features = joblib.load("E:\Coding\SIH\shubham - Final_round\stacking1_selected_features.pkl")
            # model = joblib.load("E:\Coding\SIH\shubham - Final_round\stacking1_total_stream_model.pkl")

            # df = pd.DataFrame(features, index=[0])
            # print(df)

            # X = df.iloc[:, selected_features]
            # y_pred = model.predict(X)

            # print(y_pred)
            return redirect(url_for('result', ciphertext=processed_text))
        else:
            flash('No text provided!', 'error')
            return redirect(url_for('upload'))
    return render_template('upload.html')

@app.route('/result')
def result():
    ciphertext = request.args.get('ciphertext')
    mydict = {}

    if (len(ciphertext) <= 256):
        mydict['MD5'] = predictor(ciphertext, "models/MD5/_MD5_tokenizer.joblib", "models/MD5/same_key_.MD5.keras")
        mydict['Blake'] = predictor(ciphertext, "models/Blake/_Blake_tokenizer.joblib", "models/Blake/same_key_.Blake.keras")
        mydict['Keccak'] = predictor(ciphertext, "models/Keccak/_Keccak_tokenizer.joblib", "models/Keccak/same_key_keccak.keras")
        mydict['SHA-256'] = predictor(ciphertext, "models/SHA-256/_SHA256_tokenizer.joblib", "models/SHA-256/same_key_.SHA256.keras")
        mydict['KMAC-256'] = predictor(ciphertext, "models/KMAC-256/_KMAc256_tokenizer.joblib", "models/KMAC-256/same_key_KMAC256.h5")

    mydict['ARC4'] = predictor(ciphertext, "models/ARC4/same_ktoeknizer_arc4.joblib", "models/ARC4/same_key_combined_arc4.keras")
    mydict['Blowfish'] = predictor(ciphertext, "models/Blowfish/same_key_tokenizer_combined_Blowfish.joblib", "models/Blowfish/same_key_combined_Blowfish.keras")
    mydict['CAST-128'] = predictor(ciphertext, "models/CAST-128/same_key_tokenizer_combined_cast-128.joblib", "models/CAST-128/same_key_combined_Cast-128.keras")
    mydict['ChaCha20'] = predictor(ciphertext, "models/ChaCha20/same_ktoeknizer_CHA20.joblib", "models/ChaCha20/same_key_combined_CHA20.h5")
    mydict['ECC'] = predictor(ciphertext, "models/ECC/same_key_tokenizer_combined_ecc-128.joblib", "models/ECC/same_key_combined_ecc-128.keras")
    mydict['Salsa20'] = predictor(ciphertext, "models/Salsa20/same_ktoeknizer_salsa20.joblib", "models/Salsa20/same_key_combined_salsa20.keras")
    mydict['AES-128'] = predictor(ciphertext, "models/AES-128/same_key_tokenizer_combined_AES-128.joblib", "models/AES-128/same_key_combined_AES-128.h5")
    mydict['Triple-DES'] = predictor(ciphertext, "models/Triple-DES/same_key_tokenizer_combined_Triple-DES.joblib", "models/Triple-DES/same_key_combined_Triple-DES.keras")

    sorted_dict = dict(sorted(mydict.items(), key=lambda item: item[1], reverse=True))

    # Split the dictionary into top 3 entries and the rest
    top_3 = list(sorted_dict.items())[:3]

    return render_template('result.html',top_3_predictions=top_3, sorted_dict=sorted_dict)
    

@app.route('/features')
def features():
    # Pass the 'features' dictionary to the 'features.html' template
    return render_template('features.html', features=session['features'])

# New API endpoint to serve feature data dynamically
@app.route('/api/feature-data', methods=['GET'])
def get_feature_data():
    features = session.get('features', {})
    if not features:
        return jsonify({"error": "No features available"}), 400
    return jsonify(features)

if __name__ == '__main__':
    app.run(debug=True)
