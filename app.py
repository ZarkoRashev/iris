from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load the model from disk
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        prediction = model.predict(np.array(data).reshape(1, -1))
        return jsonify({'prediction': int(prediction[0])})
    except:
        return jsonify({'error': 'Invalid data. Please check your input and try again.'})
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is running!'}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)