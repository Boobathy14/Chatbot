from flask import Flask, render_template,request
import pickle
from tensorflow import keras
import numpy as np
import json
from colorama import Fore, Style, Back
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        value = request.form['value']

        with open("intents.json") as file:
            data = json.load(file)

        model = keras.models.load_model('chat_model')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        max_len = 20

        while True:

            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([value]),
                                                                                 truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])

            for i in data['intents']:
                if i['tag'] == tag:
                    output = np.random.choice(i['responses'])

            return render_template('home.html',prediction_text='Chat-Bot: {}'.format(output))

    return render_template('home.html')


if __name__ == '__main__':
   app.run(debug=True)