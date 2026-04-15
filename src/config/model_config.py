from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor


MODELS = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(objective="reg:squarederror"),
    "AdaBoost Regressor": AdaBoostRegressor(),
}


PARAMS = {

    "Linear Regression": {},

    "Lasso":{
        "alpha":[0.01, 0.1, 1, 10]
    },

    "Ridge":{
        "alpha":[0.01, 0.1, 1, 10]
    },

    "K-Neighbors Regressor":{
        "n_neighbors":[3,5,7,9,11],
        "weights":["uniform","distance"],
        "algorithm":['ball_tree', 'kd_tree', 'auto']
    },

    "Decision Tree":{
        "criterion":["squared_error","friedman_mse"],
        "min_samples_split":[2,5,10,20],
        "min_samples_leaf":[1,2,5,10],
        "max_depth":[None,5,10,20]
    },

    "Random Forest Regressor":{
        "n_estimators":[50,100,200],
        "max_depth":[None,5,10,20],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4],
        "max_features":["sqrt","log2"]
    },

    "XGBRegressor":{
        "n_estimators":[50,100,200],
        "learning_rate":[0.01,0.05,0.1],
        "max_depth":[3,5,7],
        "subsample":[0.6,0.8,1.0],
        "colsample_bytree":[0.6,0.8,1.0]
    },

    "AdaBoost Regressor":{
        "n_estimators":[50,100,200],
        "learning_rate":[0.01,0.05,0.1,1]
    }
}