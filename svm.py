import numpy as np
class SVM:
    
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