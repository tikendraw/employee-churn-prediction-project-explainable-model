import numpy as np
import joblib
import streamlit as st
import pandas as pd
import toml
from utils.utils import explain
import webbrowser

st.set_option("deprecation.showPyplotGlobalUse", False)


input_file_name = "./config.toml"
with open(input_file_name) as toml_file:
    config = toml.load(toml_file)

column_transformer = joblib.load("./model/column_transformer.joblib")


readme = (
    "https://github.com/tikendraw/employee-churn-prediction-project/blob/main/README.md"
)
notebook_url = "https://github.com/tikendraw/employee-churn-prediction-project/blob/main/Churn_notebook.ipynb"
github_repo = "https://github.com/tikendraw/employee-churn-prediction-project"


def header():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Github"):
            webbrowser.open_new_tab(github_repo)
    with col2:
        if st.button("Notebook"):
            webbrowser.open_new_tab(notebook_url)
    with col3:
        if st.button("Readme"):
            webbrowser.open_new_tab(readme)


# Train the logistic regression model
model = joblib.load("./model/finalxgbclassifier.joblib")

# Create the Streamlit app
st.title("Churn Prediction App")
st.write("Enter customer details to predict churn")
header()


def cap(x):
    """Capitalizes strings in iterables"""
    return map(lambda x: str(x).capitalize(), x)


dropdown = config["categories_with_values"]

with st.form(key="columns_in_form"):
    name = st.text_input("Name")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", cap(dropdown["gender"]))
    with c2:
        seniorcitizen = st.selectbox("Senior Citizen", cap(dropdown["seniorcitizen"]))
    with c3:
        partner = st.selectbox("Partner", cap(dropdown["partner"]))

    c4, c5, c6 = st.columns(3)
    with c4:
        dependents = st.selectbox("Dependents", cap(dropdown["dependents"]))
    with c5:
        phoneservice = st.selectbox("Phone Service", cap(dropdown["phoneservice"]))
    with c6:
        multiplelines = st.selectbox("Multiple Lines", cap(dropdown["multiplelines"]))

    c7, c8, c9 = st.columns(3)
    with c7:
        internetservice = st.selectbox(
            "Internet Service", cap(dropdown["internetservice"])
        )
    with c8:
        onlinesecurity = st.selectbox(
            "Online Security", cap(dropdown["onlinesecurity"])
        )
    with c9:
        onlinebackup = st.selectbox("Online Backup", cap(dropdown["onlinebackup"]))

    c10, c11, c12 = st.columns(3)
    with c10:
        deviceprotection = st.selectbox(
            "Device Protection", cap(dropdown["deviceprotection"])
        )
    with c11:
        techsupport = st.selectbox("Tech Support", cap(dropdown["techsupport"]))
    with c12:
        streamingtv = st.selectbox("Streaming TV", cap(dropdown["streamingtv"]))

    c13, c14, c15 = st.columns(3)
    with c13:
        streamingmovies = st.selectbox(
            "Streaming Movies", cap(dropdown["streamingmovies"])
        )
    with c14:
        contract = st.selectbox("Contract", cap(dropdown["contract"]))
    with c15:
        paperlessbilling = st.selectbox(
            "Paperless Billing", cap(dropdown["paperlessbilling"])
        )

    c16, c17, c18 = st.columns(3)
    with c16:
        paymentmethod = st.selectbox("Payment Method", cap(dropdown["paymentmethod"]))
    with c17:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=36)
    with c18:
        monthlycharges = st.number_input(
            "Monthly Charges ($)", min_value=0, max_value=200, value=50
        )

    totalcharges = st.number_input(
        "Total Charges ($)", min_value=0, max_value=10000, value=500
    )

    submitButton = st.form_submit_button(label="Calculate")

# Define input fields

data = {
    "gender": gender,
    "seniorcitizen": seniorcitizen,
    "partner": partner,
    "dependents": dependents,
    "phoneservice": phoneservice,
    "multiplelines": multiplelines,
    "internetservice": internetservice,
    "onlinesecurity": onlinesecurity,
    "onlinebackup": onlinebackup,
    "deviceprotection": deviceprotection,
    "techsupport": techsupport,
    "streamingtv": streamingtv,
    "streamingmovies": streamingmovies,
    "contract": contract,
    "paperlessbilling": paperlessbilling,
    "paymentmethod": paymentmethod,
    "tenure": tenure,
    "monthlycharges": monthlycharges,
    "totalcharges": totalcharges,
}


# Define a function for preprocessing the input data
def preprocess(data):
    # Convert input data into a pandas DataFrame with a new index
    data = pd.DataFrame(data, index=np.arange(len(data)))

    # Convert specified columns to lowercase
    for col in dropdown.keys():
        data[col] = data[col].str.lower()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Apply a column transformation step
    data = column_transformer.transform(data)

    # Return the preprocessed data
    return data


# Check if the submit button has been clicked
if submitButton:
    # Preprocess the input data using the above function
    x = preprocess(data)

    # Use the trained model to make a prediction on the preprocessed data
    result = model.predict(x)

    # Generate a message for the user based on the prediction result
    if result == 0:
        # If the result is 0, display a message indicating that the user will not be churning
        st.markdown(
            f'<p align="center" style="color:#369af7;font-size:24px;border-radius:2%;"><b>{name}! you will be Not Churning.</b></p>',
            unsafe_allow_html=True,
        )
    else:
        # If the result is not 0, display a message indicating that the user will be churning
        st.markdown(
            f'<p  align="center" style="color:#f73664;font-size:24px;border-radius:2%;"><b>{name}! you will be Churning.</b></p>',
            unsafe_allow_html=True,
        )

    # Display the churn probability
    st.markdown(
        f'<p align="center" style="color:#2AC153;font-size:24px;border-radius:2%;"><b>Churn Probability: {model.predict_proba(x)[0][1]}</b></p>',
        unsafe_allow_html=True,
    )

    # Call an explain function with the preprocessed data, trained model, and column transformer object
    explain(x, model, column_transformer)
