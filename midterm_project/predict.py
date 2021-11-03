from flask import Flask
import pickle
from flask import request
from flask import jsonify
import xgboost as xgb
import pandas as pd
import numpy as np


with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('imputer.bin', 'rb') as f_in:
    imputer = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])

def predict():
	employee = request.get_json()
	df_employee = pd.DataFrame(employee, 
                  index=[0], 
                  columns=['city','city_development_index','gender','relevent_experience','enrolled_university','education_level','major_discipline', 'experience', 'company_size', 'company_type',
                          'last_new_job', 'training_hours'])

	# Data Cleaning
	experience_values = { '>20' : '21', '<1' : '0'}
	df_employee.experience = df_employee.experience.replace(experience_values)
	df_employee['experience'] = pd.to_numeric(df_employee['experience'])

	last_new_job_values = {'>4' : '4+'}
	df_employee.last_new_job = df_employee.last_new_job.replace(last_new_job_values)

	company_size_values = { '10/49' : '10-49', '<10' : '0-10'}
	df_employee.company_size = df_employee.company_size.replace(company_size_values)

	#apply imputer
	X = imputer.transform(df_employee)
	
	#apply encoding categorical features
	X = pd.DataFrame(X, columns = df_employee.columns)
	X_dict = X.to_dict(orient='records')
	X = dv.transform(X_dict)
	
	#convert to DMatrix
	dX = xgb.DMatrix(X, feature_names=dv.get_feature_names())

	#predict
	y_pred = model.predict(dX)[0]
	looking_for_a_job_change = y_pred >= 0.3

	result = {
		'looking_for_a_job_change_probability' : np.float64(y_pred),
		'looking_for_a_job_change' : bool(looking_for_a_job_change)
	}

	return jsonify(result)


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=9696)