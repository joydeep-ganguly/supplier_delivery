import pandas as pd
import numpy as np
import pickle
#from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request

model = pickle.load(open('delivery_promptness_rf.pkl','rb'))
cmap = pickle.load(open('workerselection_colmapping.pkl','rb'))
lmap = pickle.load(open('workerselection_mapping.pkl','rb'))
ss = pickle.load(open('worker_selection_scaled.pkl','rb'))
prediction_description = {0 : 'Early', 1 : 'Late', 2 : 'On time'}
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='GET':
        metal_list = cmap['mtcode'].tolist()
        carret_list = cmap['carret'].tolist()
        designtype_list = cmap['xrx_sam'].tolist()
        return render_template('index.html',carret_list=carret_list,designtype_list=designtype_list,metal_list=metal_list)
    else:
        #ss = StandardScaler()
        form_inputs = pd.DataFrame(request.form.to_dict(),index=[0])
        
        form_inputs['mtcode'] = form_inputs['mtcode'].apply(lambda x: lmap[x])
        form_inputs['itemcode'] = form_inputs['itemcode'].apply(lambda x: lmap[x])
        form_inputs['carret'] = form_inputs['carret'].apply(lambda x: lmap[x])
        form_inputs['xrx_sam'] = form_inputs['xrx_sam'].apply(lambda x: lmap[x])
        
        form_inputs = ss.transform(form_inputs)
        prediction = model.predict(form_inputs.astype('float'))
        result= "Order will be delivered - " + str(prediction_description[prediction[0]])
        return render_template('result.html', prediction_text=result)
        #return str(prediction)
        #return request.form.to_dict()
        #return form_inputs.to_html()

if __name__== '__main__':
    app.run(debug=True)