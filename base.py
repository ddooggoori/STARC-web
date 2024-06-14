import pandas as pd
import numpy as np
import joblib

def BMI_calculator(data):
    for index, row in data.iterrows():
        h = row['Height']
        w = row['Weight']
        
        data.loc[index, 'BMI'] = (w / (h ** 2)) * 10000
    
    return data


def final_set_maker(data):
    ator = [10, 20, 40, 80]
    pita = [2, 4]
    pra = [10, 20, 40]
    rosu = [5, 10, 20]
    sim = [20, 40]

    lists = [ator, pita, pra, rosu, sim]
    statin_name = ['Atorvastatin', 'Pitavastatin', 'Pravastatin', 'Rosuvastatin', 'Simvastatin']

    statin_sample = []
    for index, a in enumerate(lists):
        for i in range(len(a)):
            row = [0] * len(lists)
            row[index] = a[i]
            statin_sample.append(row)

    statin_sample = pd.DataFrame(statin_sample, columns=statin_name)
    statin_sample = pd.DataFrame(np.tile(statin_sample, (len(data), 1)), columns=statin_name)

    statin = statin_sample.melt(var_name='Statin', value_name='Dose')
    statin = statin[statin.Dose != 0].reset_index(drop = True)

    data_replica = pd.concat([data]* len(statin_sample)).reset_index(drop = True)

    final_set = pd.concat([data_replica, statin_sample], axis = 1)
    
    return final_set, statin


def Scaling_for_validation(data1, *args, path):
    original_variable = data1.columns

    binary = [
    'Proteinuria_3.0', 'Sex', 'Smoking', 'CKD', 
    'multiple_CV_risk', 'DM', 'Proteinuria_4.0', 
    'TOD', 'LVH_ECG', 'Drinking'
    ]
            
    data1_binary = data1[binary].reset_index(drop = True)
    
    data1_conti = data1.drop(binary, axis = 1).reset_index(drop = True)
    
    loaded_scaler = joblib.load(path)
            
    scaler = loaded_scaler
    data1_conti = pd.DataFrame(scaler.transform(data1_conti), columns=data1_conti.columns)
    
    data1_scaled = pd.concat([data1_conti, data1_binary], axis =1)[original_variable]
    
    scaled_args = list()
    if args:
        for data in args:
            data_binary = data[binary].reset_index(drop = True)
            data_conti = data.drop(binary, axis = 1).reset_index(drop = True)
            
            data_conti = pd.DataFrame(scaler.transform(data_conti), columns=data_conti.columns)
            
            data_scaled = pd.concat([data_conti, data_binary], axis =1)[original_variable]
            
            scaled_args.append(data_scaled)            
    
    if args:
        return [data1_scaled, *scaled_args]
    else:
        return data1_scaled
    
    
def Results(final_df, statin):    
    
    model = joblib.load(r'utils/STARC_model.pkl')
    predic_proba = model.predict_proba(final_df)[:, 1] * 100
    predic_proba = pd.DataFrame(predic_proba, columns = ['Predictions'])
    results = pd.concat([statin, predic_proba], axis = 1)
    
    # 순서에 따라 데이터프레임 정렬
    statin_order = ['Rosuvastatin', 'Atorvastatin', 'Pravastatin', 'Pitavastatin', 'Simvastatin']
    results['Statin'] = pd.Categorical(results['Statin'], categories=statin_order, ordered=True)
    results = results.sort_values('Statin').reset_index(drop = True)
    results['Predictions'] = round(results['Predictions'], 1)
    results.columns = ['Recommended Statin', 'Recommended Dosage', 'Success Rate']
    results['Success Rate'] = results['Success Rate'].astype(str) + '%'

    grouped = results.groupby('Recommended Statin')
    results['Recommended Statin'] = results.apply(lambda row: row['Recommended Statin'] if row.name in grouped.head(1).index else '', axis=1)
    results = results.set_index('Recommended Statin')
    
    return results
