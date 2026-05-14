from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("loan_data.csv")

# -------------------------------
# ENCODING
# -------------------------------
le_income = LabelEncoder()
le_credit = LabelEncoder()
le_emp = LabelEncoder()
le_result = LabelEncoder()

data['Income'] = le_income.fit_transform(data['Income'])
data['Credit'] = le_credit.fit_transform(data['Credit'])
data['Employment'] = le_emp.fit_transform(data['Employment'])
data['Result'] = le_result.fit_transform(data['Result'])

# -------------------------------
# TRAIN MODEL (ID3)
# -------------------------------
model = DecisionTreeClassifier(criterion='entropy')

X = data[['Income','Credit','Employment']]
y = data['Result']

model.fit(X, y)

# -------------------------------
# HOME PAGE
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# PREDICTION
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    income = request.form['income']
    credit_score = request.form['credit_score']
    employment = request.form['employment']

    # Convert credit score → Good/Bad
    if credit_score and int(credit_score) >= 750:
        credit = "Good"
    else:
        credit = "Bad"

    # Encode
    i = le_income.transform([income])[0]
    c = le_credit.transform([credit])[0]
    e = le_emp.transform([employment])[0]

    test = pd.DataFrame([[i, c, e]], columns=['Income','Credit','Employment'])

    prediction = model.predict(test)
    result = le_result.inverse_transform(prediction)[0]

    return render_template(
        'index.html',
        prediction_text=f"Loan Status: {result}",
        income=income,
        credit_score=credit_score,
        employment=employment
    )

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)