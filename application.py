import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request

model = pickle.load(open('delivery_promptness_rf.pkl','rb'))
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    else:
        ss = StandardScaler()
        form_inputs = pd.DataFrame(request.form.to_dict(),index=[0])
        form_inputs = ss.fit_transform(form_inputs)
        prediction = model.predict(form_inputs.astype('float'))
        return str(prediction)
        #return request.form.to_dict()

if __name__== '__main__':
    app.run(debug=True)