import numpy as np
from svm import SVM 
from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt




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
    print("\n entete du dataframe \n")
    print(df.head())

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


def visualize_svm(clf,X,y):
    #donne la valeur de l'hyperplan sur les deux dimentions choisie
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    #tracer le nuage de point
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:,1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    #permet de tracer la droite du discriminant( svm)
    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    #permet de tracer les droites paralelles au discriminant qui traversen les vecteurs de supports
    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    #permet de dessiner les droites precedement cités
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])

    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()
    return 0

    
def main():
    train_X,train_Y,test_X,test_Y = get_data()

    svm = SVM([0,0,0,0,0,0,0,0,0,0,0,0,0],0,1)
    svm2 = SVM([0,0,0,0,0,0,0,0,0,0,0,0,0],0,0.01)   #regularisation parameter
    
    print("\nModel 1 : \n")
    print("cout global avant entrainement =" , svm.global_cost(train_X,train_Y))
    svm.train(train_X,train_Y)
    print("exactitude= ",svm.test_success_rate(test_X,test_Y))
    print("rappel= ",svm.rappel(test_X,test_Y))
    print("precision = ",svm.precision(test_X,test_Y))
    print("cout global apres entrainement =" , svm.global_cost(train_X,train_Y))
    visualize_svm(svm,train_X,train_Y)

    print("\nModel 2 : \n")
    print("cout global avant entrainement =" , svm2.global_cost(train_X,train_Y))
    svm2.train(train_X,train_Y)
    print("exactitude= ",svm2.test_success_rate(test_X,test_Y))
    print("rappel= ",svm2.rappel(test_X,test_Y))
    print("precision = ",svm2.precision(test_X,test_Y))
    print("cout global apres entrainement =" , svm2.global_cost(train_X,train_Y))
    visualize_svm(svm2,train_X,train_Y)

    print("\nValidation croisé:\n")
    cross_validation(svm , svm2 , train_X, train_Y)

    


if __name__ == "__main__":
    main()