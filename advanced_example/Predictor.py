import warnings
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot as plt


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

housing = pd.read_csv('input_data/housing_train.csv')
y = np.log(housing.v_SalePrice)
housing = housing.drop('v_SalePrice', axis=1)
housing.drop('v_Utilities', axis=1, inplace=True)

BasementQualityDict = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
BasementExposureDict = {'Gd': 8, 'Av': 6, 'Mn': 4, 'No': 2, 'NA': 0}
BasementFinishType1Dict = {'GLQ': 12, 'ALQ': 10, 'BLQ': 8, 'Rec': 6, 'LwQ': 4, 'Unf': 2, 'NA': 0}
CentralAirDict = {'N': 0, 'Y': 2}
GarageFinishDict = {'Fin': 6, 'RFn': 4, 'Unf': 2, 'NA': 0}
FunctionalDict = {'Typ': 14, 'Min1': 12, 'Min2': 10, 'Mod': 8, 'Maj1': 6, 'Maj2': 4, 'Sev': 2, 'Sal': 0}
LandSlopeDict = {'Gtl': 4, 'Mod': 2, 'Sev': 0}
FenceDict = {'GdPrv': 8, 'MnPrv': 6, 'GdWo': 4, 'MnWw': 2, 'NA': 0}
StreetDict = {'Grvl': 0, 'Pave': 2}
PavedDriveDict = {'Y': 4, 'P': 2, 'N': 0}
ExteriorQualityDict = {'Ex': 10, 'Gd': 8, 'TA': 6, 'Fa': 4, 'Po': 2, 'NA': 0}
LotShapeDict = {'Reg': 6, 'IR1': 4, 'IR2': 2, 'IR3': 0}

housing.v_Bsmt_Exposure = housing.v_Bsmt_Exposure.map(BasementExposureDict)
housing.v_BsmtFin_Type_1 = housing.v_BsmtFin_Type_1.map(BasementFinishType1Dict)
housing.v_BsmtFin_Type_2 = housing.v_BsmtFin_Type_2.map(BasementFinishType1Dict)
housing.v_Pool_QC = housing.v_Pool_QC.map(BasementQualityDict)
housing.v_Heating_QC = housing.v_Heating_QC.map(ExteriorQualityDict)
housing.v_Central_Air = housing.v_Central_Air.map(CentralAirDict)
housing.v_Garage_Finish = housing.v_Garage_Finish.map(GarageFinishDict)
housing.v_Bsmt_Qual = housing.v_Bsmt_Qual.map(BasementQualityDict)
housing.v_Bsmt_Cond = housing.v_Bsmt_Cond.map(BasementQualityDict)
housing.v_Garage_Qual = housing.v_Garage_Qual.map(BasementQualityDict)
housing.v_Functional = housing.v_Functional.map(FunctionalDict)
housing.v_Street = housing.v_Street.map(StreetDict)
housing.v_Fence = housing.v_Fence.map(FenceDict)
housing.v_Land_Slope = housing.v_Land_Slope.map(LandSlopeDict)
housing.v_Exter_Qual = housing.v_Exter_Qual.map(ExteriorQualityDict)
housing.v_Exter_Cond = housing.v_Exter_Cond.map(ExteriorQualityDict)
housing.v_Kitchen_Qual = housing.v_Kitchen_Qual.map(ExteriorQualityDict)
housing.v_Fireplace_Qu = housing.v_Fireplace_Qu.map(ExteriorQualityDict)
housing.v_Garage_Cond = housing.v_Garage_Cond.map(ExteriorQualityDict)
housing.v_Paved_Drive = housing.v_Paved_Drive.map(PavedDriveDict)
housing.v_Lot_Shape = housing.v_Lot_Shape.map(LotShapeDict)

housing['HeatingTimesPool'] = housing.v_Heating_QC * housing.v_Pool_QC
housing['QualityAndCondition'] = (housing.v_Exter_Qual + housing.v_Fireplace_Qu + housing.v_Garage_Qual
                                  + housing.v_Kitchen_Qual + housing.v_Bsmt_Qual + housing.v_Exter_Cond
                                  + housing.v_Bsmt_Cond + housing.v_Garage_Cond + housing.v_Overall_Cond)
housing['GarageAreaPerCar'] = housing.v_Garage_Area / housing.v_Garage_Cars

rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(housing, y, random_state=rng, train_size=0.8)

# X_train.drop("parcel", axis=1, inplace=True)

cat_cols = ['v_MS_Zoning', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour', 'v_Utilities',
            'v_Land_Slope', 'v_Neighborhood', 'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
            'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd', 'v_Mas_Vnr_Type', 'v_Exter_Qual',
            'v_Exter_Cond', 'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure', 'v_BsmtFin_Type_1',
            'v_BsmtFin_Type_2', 'v_Heating', 'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_Kitchen_Qual',
            'v_Functional', 'v_Fireplace_Qu', 'v_Garage_Type', 'v_Garage_Finish', 'v_Garage_Qual', 'v_Garage_Cond',
            'v_Paved_Drive', 'v_Pool_QC', 'v_Fence', 'v_Misc_Feature', 'v_Sale_Type', 'v_Sale_Condition', 'v_Lot_Config'
            ]
numeric_cols = ['v_MS_SubClass', 'v_Lot_Frontage', 'v_Lot_Area', 'v_Overall_Qual', 'v_Overall_Cond',
                'v_Year_Built', 'v_Year_Remod/Add', 'v_Mas_Vnr_Area', 'v_BsmtFin_SF_1', 'v_BsmtFin_SF_2',
                'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_1st_Flr_SF', 'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF',
                'v_Gr_Liv_Area', 'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
                'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_TotRms_AbvGrd', 'v_Fireplaces', 'v_Garage_Yr_Blt',
                'v_Garage_Cars', 'v_Garage_Area', 'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch',
                'v_3Ssn_Porch', 'v_Screen_Porch', 'v_Pool_Area', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold',
                'v_Garage_Cars', 'QualityAndCondition', 'GarageAreaPerCar', 'v_Fence', 'v_Street', 'v_Functional',
                'v_Garage_Finish', 'v_Central_Air', 'v_BsmtFin_Type_2', 'v_BsmtFin_Type_1', 'v_Bsmt_Exposure',
                'v_Pool_QC', 'v_Garage_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Qual', 'v_Lot_Shape', 'v_Paved_Drive',
                'v_Garage_Cond', 'HeatingTimesPool', 'v_Fireplace_Qu', 'v_Kitchen_Qual', 'v_Heating_QC',
                'v_Exter_Cond', 'v_Exter_Qual', 'v_Garage_Cond'
                ]

numeric_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_pipe = make_pipeline(OneHotEncoder())

preproc_pipe = ColumnTransformer(
    [  # arg 1 of ColumnTransformer is a list, so this starts the list
        # a tuple for the numerical vars: name, pipe, which vars to apply to
        ("num_impute", numeric_pipe, numeric_cols),
        # a tuple for the categorical vars: name, pipe, which vars to apply to
        ("cat_trans", cat_pipe, ['v_Alley'])
    ]
    , remainder='drop'
)

pipe = make_pipeline(preproc_pipe, linear_model.SGDRegressor(random_state=rng))
parameters = {'sgdregressor__alpha': [0.24546, 0.24547, 0.24548, 0.24549]}
sgdregressor__alpha_grid_search = GridSearchCV(estimator=pipe,
                           param_grid=parameters,
                           cv=10
                           )
sgdregressor__alpha_grid_search.fit(X_train, y_train)
SGDRegressor_pred = sgdregressor__alpha_grid_search.best_estimator_.predict(X_test)

# Best params were [0.0152, 0.0153, 0.0154]
elastic_alpha_param_list = [0.0150, 0.0151, 0.0152, 0.0153, 0.0154, 0.0155, 0.0156, 0.0157]
elastic_l1_ratio_param_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

pipe = make_pipeline(preproc_pipe, linear_model.ElasticNet(random_state=rng))
parameters = {'elasticnet__alpha': elastic_alpha_param_list, 'elasticnet__l1_ratio': elastic_l1_ratio_param_list}
elasticnet__alpha_grid_search = GridSearchCV(estimator=pipe,
                           param_grid=parameters,
                           cv=10
                           )
elasticnet__alpha_grid_search.fit(X_train, y_train)
ElasticNet_pred = elasticnet__alpha_grid_search.best_estimator_.predict(X_test)


SGDRegressor_pred = SGDRegressor_pred.tolist()
ElasticNet_pred = ElasticNet_pred.tolist()
df = pd.DataFrame(list(zip(SGDRegressor_pred, ElasticNet_pred)), columns=['sgd', 'elast'])
df['avg'] = df.mean(axis=1)
print(r2_score(y_test, df['avg']))


cv_results = elasticnet__alpha_grid_search.cv_results_
grid_param_1 = elastic_alpha_param_list
grid_param_2 = elastic_l1_ratio_param_list
name_param_1 = 'alpha'
name_param_2 = 'L1 ratio'

# Get Test Scores Mean and std for each grid search
scores_mean = cv_results['mean_test_score']
scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

scores_sd = cv_results['std_test_score']
scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

# Plot Grid search scores
_, ax = plt.subplots(1, 1)

# Param1 is the X-axis, Param 2 is represented as a different curve (color line)
for idx, val in enumerate(grid_param_2):
    ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
ax.set_xlabel(name_param_1, fontsize=16)
ax.set_ylabel('CV Average Score', fontsize=16)
ax.legend(loc="best", fontsize=15)
ax.grid('on')
plt.show()  # will display fig1 and fig2 in different windows
exit(8)