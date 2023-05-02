import numpy as np
import scipy as sp

class LinearRegressionGD:
    """Essa classe efetua os 3 tipos mais comuns de Gradiente Descendente para uma Regressao Linear.
        - Batch GD: efetua o gradiente para o conjunto total de dados. **batch = True
        - Mini Batch GD: efetua o gradiente descendente para cada parcela do conjunto. **batch = False, batch_size>1
        - GD Estocastico: efetua o gradiente descendente para cada amostra do conjunto de dados. **batch = False, batch_size = 1
    """
    
    def __init__(self, learning_rate = 0.001, max_iter=1000, batch=True, batch_size=3, tol=1e-8, verbose=True):
        self.alpha = learning_rate       # taxa de aprendizado
        self.max_iter = max_iter         # limite maximo de iteracoes
        self.batch = batch               # se True, computa o GD em batch
        self.batch_size = batch_size     # tamanho da parcela para efetuar o GD, se batch_size=1 >> GD Estocastico
        self.scores_log = []             # usaremos essa lista para guardar a pontuacao para cada iteracao
        self.tol = tol                   # Tolerancia para o tamanho da derivada,
                                         #se for muito pequena, a mudanca nos pesos e vies sera insignificante

    def fit(self, X, y, verbose=True):
        """A funcao "fit" e usada para computar os pesos e o vies, seguindo o algoritmo do Gradiente Descendente."""
        # inicio os pesos e o vies (coeficientes e intercepcao), em 0.
        self.weights = np.zeros((X.shape[1])).astype("float64")
        self.bias = 0.
        n_samples = X.shape[0]
        
        X = X.astype("float64")
        y = y.astype("float64")
        
        # caso escolha computar o gradiente descendente em "batch", o tamanho da parcela do conjunto de dados
        # recebe o tamanho do conjunto em si
        if self.batch: 
            self.batch_size = n_samples
        
        #inicio a iteracao, ate atingir a tolerancia ou o max_iter escolhido
        for it in range(self.max_iter): 
            # a variavel "start" e uma ajudante para parcelar o conjunto de dados
            start=0
            
            # inicio a iterar sobre o conjunto de dados 
            for i in range(1, n_samples, self.batch_size):
                X_batch = X[start : self.batch_size + i]
                y_batch = y[start : self.batch_size + i]
                
                #apos retirar uma parcela do conjunto de amostra e de rotulos
                #prevejo o resultado usando a funcao linear
                y_pred = self.predict(X_batch)
                
                #apos a previsao, calcula-se a funcao "loss",
                #que determina a distancia de cada previsao ao seu devido rotulo real
                loss = self.loss(y_batch, y_pred)
                
                #com a funcao "loss" determinada,
                #calcula-se agora as derivadas dos pesos e do vies em relacao a "loss"
                dw = -np.dot(X_batch.T , loss) #derivada dos pesos 
                db = -np.sum(loss) #derivada do vies
                
                #com as derivadas, posso computar os novos pesos e vies
                #os novos pesos sao os pesos antigos + as suas derivadas, multiplicadas pela taxa de aprendizagem
                self.weights -= self.alpha * dw
                self.bias -= self.alpha * db
                                
                start = self.batch_size + i
            
            #adicionamos a media das diferencas ao nosso scores_log
            mae_ = np.mean(np.abs(y-y_pred))
            self.scores_log.append(mae_)
            if it%100==0:
                print(f"MAE na iteração {it}: {mae_}")

            #apos cada iteracao, se a derivada dos pesos atingir uma tolerancia minima, o algoritmo e finalizado
            if sum(abs(dw))<self.tol:
                print("Tolerancia atingida na iteracao: ", it)
                break

            
    def predict(self, X):
        """Preve o conjunto de amostras, baseados na funcao linear:
            f(X) = M+X * b
           M = pesos
           b = viez
           
           A funcao, np.dot() efetua o produto escalar do vetor de amostras e o vetor dos pesos.
           Ela e importante pois caso o vetor de amostra tenha mais de uma dimensao, uma multiplicacao simples
           retornaria um resultado errado.
        """
        return np.dot(X , self.weights) + self.bias
    
    def loss(self, y, y_pred):
        """Funcao Loss ou Funcao de Perda, ela nos define a distancia entre os valores previstos e os reais."""
        return y-y_pred