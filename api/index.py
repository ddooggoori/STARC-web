from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from base import *
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from utils import *


app = Flask(__name__, template_folder=r'E:\WORKING\STARC\templates')
app.secret_key = 'pengdori'


@app.route('/')
def home():
    return render_template('home/home.html')


@app.route('/starc', methods=['GET', 'POST'])
def starc():
    if request.method == 'POST':
        form_data = request.form
        inputs = {}
        for key, value in form_data.items():
            inputs[key] = float(value)
        
        session['inputs'] = inputs  
        
        return redirect(url_for('starc_results'))
    
    return render_template('starc/starc.html')



@app.route('/starc_results', methods=['GET', 'POST'])
def starc_results():
    inputs = session.get('inputs', {})  
    data = pd.DataFrame(inputs, index=[0])
    print(data)
    
    data = BMI_calculator(data)
    data = eGFR_calculator(data)
    data = TOD_calculator(data)
    data = MCR_calculator(data)
    data = SCORE2_calculator(data)
    
    results = SRS_results(data)

    return render_template('starc/starc_results.html', results=results)



@app.route('/similarity', methods=['GET', 'POST'])
def similarity():
    if request.method == 'POST':
        # HTML 폼에서 받은 데이터 가져오기
        drug_name = request.form['drug']
        drug_name = drug_name.lower()
        radius = int(request.form['radius'])
        nBits = int(request.form['nBits'])

        # utils.py의 Medication_preprocessing 함수 호출하여 결과 데이터프레임 생성
        results = Medication_preprocessing(drug_name, radius, nBits)        
        print(results) 

        # similarity.html에 결과 데이터프레임 전달
        return render_template('similarity/similarity.html', results=results)
    else:
        return render_template('similarity/similarity.html')



@app.route('/survey')
def survey():
    return render_template('survey/survey.html')


@app.route('/calculate')
def calculate():
    return render_template('calculate/calculate.html')



if __name__ == '__main__':
    app.debug = True
    app.run()

