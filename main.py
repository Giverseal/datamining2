



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
