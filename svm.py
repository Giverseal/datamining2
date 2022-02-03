import numpy as np
class SVM:
    
    def __init__(self, w : list = None , b : float = None 
                 ,regularisation_parameter :float = 0.01 ):
        self.__w = w
        self.__b = b
        self.__regularisation_parameter = regularisation_parameter
    
    @property
    def w(self) -> list:
        return self.__w
    
    @property
    def b(self) -> float:
        return self.__b
    
    @property
    def regularisation_parameter(self) -> float:
        return self.__regularisation_parameter

    @property
    def set_w(self, w):
        self.__w = w
    
    @property
    def set_b(self, b : float):
        self.__b = b
    
    def compute(self, x : list ) -> float:
        return np.dot(x, self.w) - self.b
    
    
    def train(self, X : list, Y : list , number_of_iterations : int = 1000 
              , learning_rate : float = 0.01 ):
        number_of_lines, number_of_columns = X.shape
        if(self.w is None):
            self.__b = 0
        self.__w = np.zeros(number_of_columns)
        print("global cost before" , self.global_cost(X,Y))
        for i in range(number_of_iterations):
            for index, x in enumerate(X):
                if( Y[index] * self.compute(x) >= 1):
                    gradientW = 2 * self.regularisation_parameter * self.w
                    gradientB = 0
                else:
                    gradientW = 2 * self.regularisation_parameter * self.w - Y[index] * x
                    gradientB = Y[index]
                self.__w =  self.w -  learning_rate * gradientW 
                self.__b =  self.b - learning_rate * gradientB 
    
    
    def predict(self, x):
        if( self.compute(x) >= 0 ):
            return 1
        else:
            return -1

    def test_success_rate(self,X,Y):    #exactitude
        number_of_lines, number_of_columns = X.shape
        success = 0
        for i,x in enumerate(X):
            if(self.predict(x) == Y[i]):
                success = success + 1; 
        return success/number_of_lines 

    def precision(self,X,Y):
        number_of_lines, number_of_columns = X.shape
        tp = 0
        tp_fp =0
        for i,x in enumerate(X):
            if(self.predict(x) == Y[i] == 1):
                tp = tp + 1; 
            if((self.predict(x) == Y[i] == 1) or ((self.predict(x) == 1)and (Y[i]== -1 ))):
                tp_fp = tp_fp + 1
        return tp/tp_fp 
    
    def rappel(self,X,Y): #rappel
        number_of_lines, number_of_columns = X.shape
        tp = 0
        tp_fn =0
        for i,x in enumerate(X):
            if(self.predict(x) == Y[i] == 1):
                tp = tp + 1; 
            if((self.predict(x) == Y[i] == 1) or ((self.predict(x) == -1)and (Y[i]== 1 ))):
                tp_fn = tp_fn + 1
        return tp/tp_fn 
    
    def cost(self, x: list, y: int) -> float:
        if( y * self.compute(x) >= 1):
            return 0
        else:
            return 1 - y* self.compute(x)
    
    def global_cost(self , X:list , Y:list)-> float:
        number_of_lines , number_of_columns = X.shape
        part1 = self.regularisation_parameter * np.dot(self.w , self.w)
        somme = 0
        for index, x in enumerate(X):
            somme = somme + self.cost(x,Y[index])
        return part1 + somme
    
    def normalized_global_cost(self , X:list , Y:list)-> float:
        number_of_lines , number_of_columns = X.shape
        part1 = self.regularisation_parameter * np.dot(self.w , self.w)
        somme = 0
        for index, x in enumerate(X):
            somme = somme + self.cost(x,Y[index])
        return part1 + (1/number_of_lines) * somme
