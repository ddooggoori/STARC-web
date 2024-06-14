from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from base import *
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from utils import *


app = Flask(__name__, template_folder=r'E:\WORKING\STARC\templates')


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



@app.route('/survey')
def survey():
    return render_template('survey/survey.html')



if __name__ == '__main__':
    app.debug = True
    app.run()

