import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

input_file = 'model_eta=0.1_max_depth=3_v2.53.bin'

with open(input_file, 'rb') as f_in: 
    dv, model, features = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
	crab_info = request.get_json()

	X = dv.transform([crab_info])
	dX = xgb.DMatrix(X, feature_names=features)
	y_pred = model.predict(dX)
	age = y_pred.round(1)

	result = {
		'crab_age': float(age)
	}

	return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9696)
