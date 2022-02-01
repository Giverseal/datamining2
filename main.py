import numpy as np
import pandas as pd



#les fonctions de pretraitement 

def convert_results(target : int)->int:
    if target == 0:
        return -1
    if target == 1:
        return 1
    else:
        raise ValueError('Mauvaise donnée: ' + target)
    
def convert_data(df:DataFrame)->list:
    r = []
    for index,line in df.iterrows():
        obs = [
            float(line['age']),
            float(line['sex']),
            float(line['cp']),
            float(line['trestbps']),
            float(line['chol']),
            float(line['fbs']),
            float(line['restecg']),
            float(line['thalach']),
            float(line['exang']),
            float(line['oldpeak']),
            float(line['slope']),
            float(line['ca']),
            float(line['thal']),
            convert_results(line['target'])
        ]
        r.append(obs)
    return r

def get_data():
    #Importation(chargement) des données depuis le .csv:
    df = pd.read_csv('dataset.csv')

    #visualisation de l'entete:
    print("entete du dataframe \n", df.head())

    #transformation du dataframe en tableau numpy pour faciliter son utilisation
    data = np.array(convert_data(df))
   

    #séparation des entrées et des sortie:
    np.random.shuffle(data)
    number_of_lines , number_of_columns = data.shape
    X = data[:,:(number_of_columns-1)]
    Y = data[:, (number_of_columns-1):]

    #Normalisation:
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    X = (X-mean) / std

    #Separation des données en partie entrainement et partie test:
    train_X = X[:(round(0.8*number_of_lines)),:]
    train_Y = Y[:(round(0.8*number_of_lines)),:]
    test_X = X[(round(0.8*number_of_lines)):,:]
    test_Y = Y[(round(0.8*number_of_lines)):,:] 

    return train_X,train_Y,test_X,test_Y 



#  validation croisée 
def cross_validation(svm: SVM , svm2 : SVM , train_X:list , train_Y: list):
    NB_VOLETS = 10
    score_mod1 = 0 
    score_mod2 = 0 
    div_index = int(0.10 * len(train_X))

    #validation croisée
    for i in range(0, NB_VOLETS):
        # calcul des indices à utiliser pour l'entrainement / validation
        validation_range = range(i * div_index, (i+1) * div_index)
        train_range = list(set(range(len(train_X))) - set(validation_range))

        svm.__w = None
        svm.__b = 0
        svm2.__w = None
        svm2.__b = 0
        #
        x_train = train_X[train_range]
        y_train = train_Y[train_range]
        x_val = train_X[validation_range]
        y_val = train_Y[validation_range]

        #
        svm.train(x_train, y_train)
        acc1 = svm.test_success_rate(x_val,y_val)
        svm2.train(x_train, y_train)
        acc2 = svm2.test_success_rate(x_val,y_val)
        
        #
        if acc1 > acc2:
            score_mod1 += 1
        elif acc2> acc1:
            score_mod2 += 1

    if score_mod1 >= score_mod2:
        print('Résultat de la validation croisée: Le modèle 1 est meilleur')
    else:
        print('Résultat de la validation croisée: Le modèle 2 est meilleur')
