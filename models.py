import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import learning_curve, SelectKBest, mutual_info_regression

import warnings
warnings.filterwarnings('ignore')




class Model:
    
    def __init__(self, df):
        
        self.df = df
        
    def select_cols(self, target, all_cols=True, col_x = None, get_dummies = True):
    
    
        if all_cols:
            self.X, self.y = self.df.drop(target, axis=1), self.df[target]
        else:
            self.X, self.y = self.df[col_x], self.df[target]

        if get_dummies == True:

            dummies = self.X.columns[(self.X.dtypes =='category') | (self.X.dtypes == object)]

            for dummy in dummies:
                col_dummies = pd.get_dummies(self.X[dummy], prefix = dummy).iloc[:, 1:]
                self.X = self.X.join(col_dummies)
                self.X = self.X.drop([dummy], axis = 1)


        
    def regression(self, model, seed=None):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=seed)
        self.estimator = model.fit(self.X_train, self.y_train)
        self.pred = model.predict(self.X_test)
        result = mean_absolute_error(self.y_test, self.pred)
        self.score_train = model.score(self.X_train, self.y_train)
        self.score_test = model.score(self.X_test, self.y_test)
        
        
        return result
    
    def clasification(self, model, seed=None):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=seed)
        self.estimator = model.fit(self.X_train, self.y_train)
        self.pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.pred)
        matrix = confusion_matrix(self.y_test, self.pred)
        self.score_train = model.score(self.X_train, self.y_train)
        self.score_test = model.score(self.X_test, self.y_test)
        
        return accuracy
    
    def cross_val(self, model, n):
        
        cv = abs(cross_val_score(model, self.X, self.y, scoring = 'neg_mean_absolute_error', cv=n, n_jobs=3))
        cv_mean = cv.mean()
        
        self.data_cv = pd.DataFrame(cv, columns=['mae'])
        
        sns.set()
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        plt.boxplot(self.data_cv['mae']);

        plt.subplot(122)
        plt.hist(self.data_cv['mae']);

        
        return self.data_cv.describe().T
    
    def optimization_model(self, model, params, seed = None):
    
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=seed)
        grid = GridSearchCV(model, param_grid=params, scoring='r2', cv=4, n_jobs=3)
        grid.fit(X_train, y_train)
        print(grid.best_score_, grid.best_params_)

        return grid.best_estimator_

    
    def monte_carlo_simulation(self,model,n):
        
        result = []
        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
            result.append(mean_absolute_error(y_test, model.fit(X_train, y_train).predict(X_test)))
        
        data_result = pd.DataFrame(result, columns=['summary']).describe()
    
        sns.set()
        plt.figure(figsize=(16,6))
        plt.subplot(121)
        plt.boxplot(result);

        plt.subplot(122)
        plt.hist(result);
    
  
    
        return data_result.T
    
    def learing_data_curve(self, model, n):

        samples, train, test = learning_curve(model, self.X, self.y, cv= 5)

        plt.plot(samples[n:], np.mean(train, axis=1)[n:])
        plt.plot(samples[n:], np.mean(test, axis=1)[n:])
        plt.show()
    
   
    def feature_selection(self, k):
    
        selector = SelectKBest(mutual_info_regression, k = k)
        selector.fit(self.X, self.y)
        values = np.round(selector.scores_,2)
        cols = self.X.columns
        self.featur_importance = pd.DataFrame(list(zip(cols, values)), 
                                              columns= ['features', 'importance']).sort_values(by='importance',ascending=False)
        
        plt.plot(values)
        plt.xticks(np.arange(len(cols)-1), list(cols))
        plt.show()
    
    
    
    def plot_residuals(self):
        
        command = input("""What plot Do You want?
        
            1) Residual Error
            2) Percentaje Residual Error
            3) Log Percentaje Residual Error
        """)
        
        if command == '1':
            
            residuals = self.y_test - self.pred
            data_residuals = pd.DataFrame(residuals)

            plt.figure(figsize=(18,6))
            plt.subplot(131)
            plt.scatter(self.y_test, residuals)

            plt.subplot(132)
            plt.hist(residuals)

            plt.subplot(133)
            plt.boxplot(residuals)
            plt.show()

            return data_residuals.describe().T
            
            
        elif command == '2':
            ap_residuals = abs(self.y_test - self.pred) / self.y_test
            data_ap_residuals = pd.DataFrame(ap_residuals)
            
            plt.figure(figsize=(18,6))
            plt.subplot(131)
            plt.scatter(self.y_test, ap_residuals)
            
            plt.subplot(132)
            plt.hist(ap_residuals)
            
            plt.subplot(133)
            plt.boxplot(ap_residuals)
            
            plt.show()
            
            return data_ap_residuals.describe().T
            
            
        elif command == '3':
            
            lap_residuals = np.log(ap_residuals)
            plt.scatter(self.y_test, lap_residuals)
            return plt.show()
        