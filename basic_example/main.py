# predictive Model

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df2 = pd.read_csv("out.csv")
df2 = df2.dropna(subset=['energy_price'])
y = df2.energy_price
y = y.iloc[1:]
housing = df2.drop('energy_price', axis=1)
housing = housing[:-1]

rng = np.random.RandomState(420)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng, train_size=0.83)

numeric_cols = ['expected_energy',
                'expected_health',
                'expected_real_estate',
                'RE_inflat_ratio',
                'energy_inflation',
                'real_estate_price',
                'actual_inflation',
                'heath_inflat_ratio',
                'health_price',
                'real_estate_inflation',
                'health_inflation',
                'energy_inflation',
                'Expected Inflation',
                'Real Risk Premium',
                'Inflation Risk Premium',
                'Expected Inflation'
                ]
numeric_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), StandardScaler())
preproc_pipe = ColumnTransformer([("num_impute", numeric_pipe, numeric_cols),],remainder='drop')
pipe = make_pipeline(preproc_pipe, linear_model.ElasticNet(random_state=rng))
param_1_List = [0.001, 0.02, 0.3, 0.4, 0.5, 0.6, 0.9]
param_2_List = [0.01, 0.02, 0.03, 0.04, 0.05, 0.6, 0.7, 0.8, 0.9]
parameters = {'elasticnet__alpha': param_1_List, 'elasticnet__l1_ratio': param_2_List}
grid_search = GridSearchCV(estimator=pipe, param_grid=parameters)
grid_search.fit(X_train, y_train)
print("The optimized parameters output of the grid search are:")
print(grid_search.best_params_)
print("The score of the optimized output of the grid search on the training data is: " + str(grid_search.best_score_))
scores = grid_search.score(X_test, y_test)
print("Using our optimized model, the R2 score on the hold out data is: " + str(scores))
predictions = grid_search.best_estimator_.predict(X_test)
