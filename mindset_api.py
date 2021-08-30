from flask import Flask,request,jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict ():
    
    feat_data = request.json
    
    df = pd.DataFrame(feat_data)
    df = df.reindex(columns=col_names)
    df = df.drop('POLICY_ID',axis=1)

    df= df.drop('DATA_TYPE',axis=1)

    df = df.drop('POLICY_INTERMEDIARY',axis=1)

    df = df.drop('VEHICLE_MODEL',axis=1)

    df = df.drop('CLAIM_AVG_ACC_ST_PRD',axis=1)

    df = df.drop('VEHICLE_MAKE',axis=1)

    df = df.drop('POLICY_CLM_N',axis=1)

    df = df.drop('POLICY_CLM_GLT_N',axis=1)

    df = df.drop('POLICY_PRV_CLM_N',axis=1)

    df = df.drop('POLICY_PRV_CLM_GLT_N',axis=1)

    df = df.drop('POLICY_BEGIN_MONTH',axis=1)

    df = df.drop('POLICY_END_MONTH',axis=1)
    
    df = df.drop('CLIENT_REGISTRATION_REGION',axis=1)
    
    df = pd.get_dummies(df,drop_first=True)

    prediction = list(model.predict(df))
    
    return jsonify({'prediction':str(prediction)})

if __name__ == '__main__':
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('col_names.pkl')
    
    
    app.run(debug=True)