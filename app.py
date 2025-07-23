import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load data and model
df_1 = pd.read_csv("first_telc.csv")
model = joblib.load("model.sav")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    try:
        # Get all form inputs
        inputs = {
            'SeniorCitizen': int(request.form['query1']),
            'MonthlyCharges': float(request.form['query2']),
            'TotalCharges': float(request.form['query3']),
            'gender': request.form['query4'],
            'Partner': request.form['query5'],
            'Dependents': request.form['query6'],
            'PhoneService': request.form['query7'],
            'MultipleLines': request.form['query8'],
            'InternetService': request.form['query9'],
            'OnlineSecurity': request.form['query10'],
            'OnlineBackup': request.form['query11'],
            'DeviceProtection': request.form['query12'],
            'TechSupport': request.form['query13'],
            'StreamingTV': request.form['query14'],
            'StreamingMovies': request.form['query15'],
            'Contract': request.form['query16'],
            'PaperlessBilling': request.form['query17'],
            'PaymentMethod': request.form['query18'],
            'tenure': int(request.form['query19'])
        }

        # Create DataFrame from inputs
        new_df = pd.DataFrame([inputs])
        
        # Process the data (without one-hot encoding numerical features)
        df_2 = pd.concat([df_1, new_df], ignore_index=True)
        
        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2['tenure'], range(1, 80, 12), right=False, labels=labels)
        
        # One-hot encode ONLY categorical variables
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'tenure_group'
        ]
        
        numerical_cols = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']
        
        # Get dummies for categoricals only
        df_categorical = pd.get_dummies(df_2[categorical_cols])
        df_numerical = df_2[numerical_cols]
        
        # Combine numerical and categorical features
        final_df = pd.concat([df_numerical, df_categorical], axis=1)
        
        # Drop the original tenure column (keep tenure_group)
        final_df.drop('tenure', axis=1, inplace=True)
        
        # Make prediction
        single = model.predict(final_df.tail(1))
        probability = model.predict_proba(final_df.tail(1))[:,1]
        
        if single == 1:
            o1 = "This customer is likely to be churned!!"
            o2 = "Confidence: {:.2f}%".format(probability[0]*100)
        else:
            o1 = "This customer is likely to continue!!"
            o2 = "Confidence: {:.2f}%".format((1-probability[0])*100)
            
        return render_template('home.html', 
                            output1=o1, 
                            output2=o2,
                            **{f'query{i}': request.form.get(f'query{i}', '') for i in range(1, 20)})
    
    except Exception as e:
        return render_template('home.html', 
                            output1=f"Error: {str(e)}", 
                            output2="Please check your inputs and try again",
                            **{f'query{i}': request.form.get(f'query{i}', '') for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)