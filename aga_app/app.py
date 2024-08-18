from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load your trained model
model = joblib.load('ckner.pkl')

# Download only necessary NLTK resources
nltk.download('punkt')

def word2features(sent, i):
    word = sent[i]
    
    features = {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sent[i - 1],
        'next_word': '' if i == len(sent) - 1 else sent[i + 1],
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def predict_entities(sentence):
    tokens = word_tokenize(sentence)
    features = [sent2features(tokens)]
    prediction = model.predict(features)[0]
    return list(zip(tokens, prediction))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    entities = predict_entities(text)
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)