import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import math

def Medication_preprocessing(target, radius=2, nBits=1024):
    
    smiles = pd.read_csv(r'E:\WORKING\STARC\utils\smiles.csv')
    
    ecfp = {'Medication': [], 'ECFP': []}
    for medication in smiles['name'].unique():
        tmp = smiles[smiles['name'] == medication]
        mol = Chem.MolFromSmiles(tmp['smiles'].values[0])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            fp_str = fp.ToBitString()
            ecfp['Medication'].append(tmp['name'].values[0])
            ecfp['ECFP'].append(fp_str)
            
    ecfp = pd.DataFrame(ecfp)
    ecfp_tmp = ecfp['ECFP'].str.split('', expand=True)
    
    ecfp = pd.concat([ecfp, ecfp_tmp], axis=1).drop(['ECFP', 0, nBits+1], axis=1)

    def tanimoto_similarity(vector1, vector2):
        intersection = np.logical_and(vector1, vector2)
        union = np.logical_or(vector1, vector2)
        return intersection.sum() / float(union.sum())
    
    tanimoto_results = {'Medication': [], 'Score': []}
    
    vector1 = ecfp[ecfp['Medication'] == target].iloc[:, 1:].values.astype(int)
       
    for medication in ecfp['Medication'].unique():
        if target != medication:
            vector2 = ecfp[ecfp['Medication'] == medication].iloc[:, 1:].values.astype(int)
            tmp = []
            for a in vector1:
                for n in vector2:
                    similarity_tmp = tanimoto_similarity(a, n)
                    tmp.append(similarity_tmp)
            
            similarity = np.mean(tmp)
            
            tanimoto_results['Medication'].append(medication)            
            tanimoto_results['Score'].append(similarity)    

    tanimoto_results = pd.DataFrame(tanimoto_results)
    tanimoto_results = tanimoto_results.sort_values(by = ['Score'], ascending=False)
    tanimoto_results = round(tanimoto_results, 3)
    
    return tanimoto_results.head(5)



def Scaling(train, *args, method = 'scaling', binary = None, exception = None, scaler = StandardScaler(), save = False):
    
    original_variable = train.columns

    if exception != None:
        train_exception = train[exception]
        train = train.drop(exception, axis = 1)

    if binary == 'all':
        if method == 'scaling':
            train_total = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.transform(train), columns=train.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns)   
    
    else:    
        if binary is None:
            binary = list()
            for i in train.columns:
                if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                    binary.append(i)
        else:
            binary = binary
        
        train_binary = train[binary].reset_index(drop=True)
        train_conti = train.drop(binary, axis=1).reset_index(drop=True)
        
        if method == 'scaling':
            train_conti = pd.DataFrame(scaler.fit_transform(train_conti), columns=train_conti.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.transform(train_conti), columns=train_conti.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.inverse_transform(train_conti), columns=train_conti.columns)    
        
        train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)
        
    if exception != None:
        train_total[exception] = train_exception
        
    train_total = train_total[original_variable]
                
    scaled_args = list()    
    if args:
        for data in args:
            if exception != None:
                data_exception = data[exception]
                data = data.drop(exception, axis = 1)

            if binary == 'all':
                data_total = pd.DataFrame(scaler.transform(data), columns=train.columns)        

            else:            
                data_binary = data[binary].reset_index(drop=True)
                data_conti = data.drop(binary, axis=1).reset_index(drop=True)

                if method == 'scaling' or method == 'apply':
                    data_conti = pd.DataFrame(scaler.transform(data_conti), columns=train_conti.columns)        
                elif method == 'inverse':
                    data_conti = pd.DataFrame(scaler.inverse_transform(data_conti), columns=train_conti.columns)   
                
                data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)
                
            if exception != None:
                data_total[exception] = data_exception
            
            data_total = data_total[original_variable]
            
            scaled_args.append(data_total)
        
    if save != False:
        joblib.dump(scaler, save + '.joblib')
    
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total
    
    
def SRS_results(data):
    
    SRS_feature =  [
    'Hba1c', 'LDL_A', 'BUN', 'multiple_CV_risk', 'TC', 'Proteinuria_4.0', 'BMI', 
    'Sex', 'TG', 'GGT', 'Weight', 'Smoking', 'Proteinuria_3.0', 'DM', 'eGFR', 'SBP', 
    'LVH_ECG', 'Glucose', 'TOD', 'Age', 'SCORE2', 'DBP', 'Height', 'HDL', 'CKD', 'Drinking',
    'Atorvastatin', 'Pitavastatin', 'Pravastatin', 'Rosuvastatin', 'Simvastatin'
    ]

    ator = [10, 20, 40, 80]
    pita = [2, 4]
    pra = [10, 20, 40]
    rosu = [5, 10, 20]
    sim = [20, 40]

    lists = [ator, pita, pra, rosu, sim]
    statin_name = ['Atorvastatin', 'Pitavastatin', 'Pravastatin', 'Rosuvastatin', 'Simvastatin']

    sum_num = sum(len(l) for l in lists)
    data_num = len(data)

    statin_sample = []

    for index, a in enumerate(lists):
        for i in range(len(a)):
            row = [0] * len(lists)
            row[index] = a[i]
            statin_sample.append(row)

    statin_sample = pd.DataFrame(statin_sample, columns=statin_name)
    statin_sample = pd.DataFrame(np.tile(statin_sample, (len(data), 1)), columns=statin_name)

    clinical_sample = pd.concat([data] * sum_num, axis=0, ignore_index=True).reset_index(drop = True)

    binary = [
        'multiple_CV_risk', 'Proteinuria_4.0',
        'Sex', 'Smoking', 'Proteinuria_3.0', 'DM',
        'LVH_ECG', 'TOD', 'CKD', 'Drinking']

    clinical_statin = pd.concat([clinical_sample, statin_sample], axis = 1)[SRS_feature]

    clinical_statin = Scaling(clinical_statin, scaler = r'E:\WORKING\STARC\utils\STARC_scaler', method = 'apply', binary = binary)

    model = joblib.load(r'E:\WORKING\STARC\utils\STARC_model.pkl')
    pred_proba = model.predict_proba(clinical_statin[SRS_feature].values)[:, 1] * 100

    predictions = pd.DataFrame(pred_proba, columns=['Success Rate'])
    predictions.reset_index(drop=True, inplace=True)
    predictions = round(predictions, 1)

    statin_df = []
    for dose in ator:
        statin_df.append({'Recommended Statin': 'Atorvastatin', 'Recommended Dose': dose})   
    for dose in pita:
        statin_df.append({'Recommended Statin': 'Pitavastatin', 'Recommended Dose': dose})
    for dose in pra:
        statin_df.append({'Recommended Statin': 'Pravastatin', 'Recommended Dose': dose})  
    for dose in rosu:
        statin_df.append({'Recommended Statin': 'Rosuvastatin', 'Recommended Dose': dose})        
    for dose in sim:
        statin_df.append({'Recommended Statin': 'Simvastatin', 'Recommended Dose': dose})

    statin_df = pd.DataFrame(statin_df)
    statin_df = pd.concat([statin_df] * data_num, axis=0, ignore_index=True).reset_index(drop = True)

    results = pd.concat([statin_df, predictions], axis = 1)

    return results


def eGFR_calculator(data):
    
    creatinine = data['CR']
    age = data['Age']
    sex = data['Sex']
    
    if data['Sex'].item() == 0:
        egfr = 186 * (creatinine ** -1.154) * (age ** -0.203) 
    
    else:
        egfr = 186 * (creatinine ** -1.154) * (age ** -0.203) * 0.742
    
    data['eGFR'] = egfr
    
    return data


def eGFR_calculator(data):
    creatinine = data['CR'].item()
    age = data['Age'].item()
    sex = data['Sex'].item()
    black = data['Black'].item()
    
    egfr = 175 * (creatinine)** -1.154 * (age ** -0.203)
    if sex == 0:
        egfr *= 0.742
    if black == 1:
        egfr *= 1.212
    
    data['eGFR'] = egfr        
    return data


    
def BMI_calculator(data):
    height = data['Height']
    weight = data['Weight']
    
    bmi = (weight / (height ** 2)) * 10000
    data['BMI'] = bmi
    return data

   
   
def TOD_calculator(data):

    if data['eGFR'].item() < 30 or data['Proteinuria_3.0'].item() == 1 or data['LVH_ECG'].item() == 1:
        data['TOD'] = 1

    else:
        data['TOD'] = 0
    
    return data


def MCR_calculator(data):
    conditions_met = 0

    if data['eGFR'].item() < 30:
        conditions_met += 1
    if data['Proteinuria_3.0'].item() == 1:
        conditions_met += 1
    if data['LVH_ECG'].item() == 1:
        conditions_met += 1
    if data['BMI'].item() > 25:
        conditions_met += 1

    if conditions_met >= 3:
        data['multiple_CV_risk'] = 1
    else:
        data['multiple_CV_risk'] = 0
    
    return data


def SCORE2_calculator(data):
    
    age = data['Age'].item()
    sbp = data['SBP'].item()
    tchol = data['TC'].item()
    hdl = data['HDL'].item()
    smoking = data['Smoking'].item()
    DM = data['DM'].item()
    sex = data['Sex'].item()    
   
    lowrisk = 0
    
    def custom_log(x):
        return 0 if x == 0 else math.log(x)
    
    if age < 70:    
        cage = (age - 60)/5 
        csbp = (sbp - 120)/20
        ctchol = (tchol/38.67) - 6
        chdl = ((hdl/38.67) - 1.3)/0.5 
        
        if sex == 0:
            beta_male = (0.3742*cage) + (0.6012*smoking) + (0.2777*csbp) + (0.6457*DM) + (0.1458*ctchol) - (0.2698*chdl) - (0.0755*cage*smoking) - (0.0255*cage*csbp) - (0.0281*cage*ctchol) + (0.0426*cage*chdl) - (0.0983*cage*DM)
            risk_male = 1 - (0.9605 ** math.exp(beta_male))
            lowrisk = 1 - (math.exp(-math.exp(-0.5699+0.7476 * custom_log(-custom_log(1 - risk_male)))))
            
        else:
            beta_female = (0.4648*cage) + (0.7744*smoking) + (0.3131*csbp) + (0.8096*DM) + (0.1002*ctchol) - (0.2606*chdl) - (0.1088*cage*smoking) - (0.0277*cage*csbp) - (0.0226*cage*ctchol) + (0.0613 *cage*chdl) - (0.1272*cage*DM)
            risk_female = 1 - (0.9776**math.exp(beta_female))
            lowrisk = 1 - math.exp(-math.exp(-0.7380+0.7019 * custom_log(-custom_log(1-risk_female))))
    
    else:    
        cage = (age - 73)
        csbp = (sbp - 150)
        ctchol = (tchol/38.67) - 6
        chdl = (hdl/38.67) - 1.4
        
        if sex == 0:
            beta_male = (0.0634*cage) + (0.3524*smoking) + (0.0094*csbp) + (0.4245*DM) + (0.0850*ctchol) - (0.3564*chdl) - (0.0247*cage*smoking) - (0.0005*cage*csbp) + (0.0073*cage*ctchol) + (0.0091*cage*chdl) - (0.0174*cage*DM)
            risk_male = 1 - (0.7576**math.exp(beta_male-0.0929))
            lowrisk = 1 - (math.exp(-math.exp(-0.61+0.89  * custom_log(-custom_log(1-risk_male)))) )
            
        else:
            beta_female = (0.0789*cage) + (0.4921*smoking) + (0.0102*csbp) + (0.6010*DM) + (0.0605*ctchol) - (0.3040*chdl) - (0.0255*cage*smoking) - (0.0004*cage*csbp) - (0.0009*cage*ctchol) + (0.0154*cage*chdl) - (0.0107*cage*DM)
            risk_female = 1 - (0.8082**math.exp(beta_female-0.229))
            lowrisk = 1 - (math.exp(-math.exp(-0.85+0.82 * custom_log(-custom_log(1-risk_female)))))
    
    data['SCORE2'] = lowrisk
    
    return data
