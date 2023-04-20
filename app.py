import pandas as pd
import numpy as np
import joblib
import toml
from flask import Flask, jsonify, request
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder , StandardScaler, PowerTransformer, LabelBinarizer, LabelEncoder
from xgboost import XGBClassifier

app = Flask(__name__)
input_file_name = "./config.toml"
with open(input_file_name) as toml_file:
    config = toml.load(toml_file)

column_transformer = joblib.load('./model/column_transformer.joblib')
model = joblib.load('./model/finalxgbclassifier.joblib')


@app.route('/', methods=['POST'])
def predict_churn():
    data = request.json
    data = pd.DataFrame(data, index=[0])

    dropdown = config['categories_with_values']
    for col in dropdown.keys():
        data[col] = data[col].str.lower()

    data = data.drop_duplicates()
    data = column_transformer.transform(data)
    result = model.predict(data)

    if result==0:
        output = {'prediction': 'Not Churning'}
    else:
        output = {'prediction': 'Churning'}

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
