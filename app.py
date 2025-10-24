from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('processed_health_data.csv')

X = df.drop(['health_risk_ord'],axis = 1)

y= df['health_risk_ord']

model = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X,y)

model.fit(X_train,y_train)

app = Flask(__name__)
CORS(app)

def predict_health_risk(input_data):
    print(input_data)
    test = pd.DataFrame([input_data])
    prediction = model.predict(test)
    print(prediction[0])


    return int(prediction[0])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Serves your HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # print(data)
        prediction = predict_health_risk(data)
        return jsonify({'prediction': prediction, 'status': 'success'})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e), 'prediction': 'Error in processing'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
