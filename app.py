from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from base import *
from utils import *


app = Flask(__name__)
app.secret_key = 'pengdori'



@app.route('/', methods=['GET', 'POST'])
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


if __name__ == '__main__':
    app.debug = True
    app.run()

