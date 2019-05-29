import pandas as pd
import setting
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    ECFP = pd.read_csv(setting.ECFP_file, index_col=0)  ##drugs fingerprint info
    # ECFP_filter = (ECFP == 0).all(axis = 0)
    # ECFP = ECFP.loc[:, ~ECFP_filter]
    # print(ECFP.shape)
    #gene_expression = pd.read_csv(setting.gene_expression_file).T

    gene_sensitivity = pd.read_csv(setting.gene_sensitivity_file)
    # cellline = {'RKO', 'OCUBM', 'NCIH520', 'UACC62', 'A375', 'T47D', 'MDAMB436', 'KPL1', 'LOVO', 'SKMEL30', 'ZR751', 'A2058',
    #  'OV90', 'SW837', 'HCT116', 'NCIH23', 'SKOV3', 'NCIH460', 'A2780', 'SW620', 'A427', 'DLD1', 'RPMI7951', 'HT144',
    #  'CAOV3', 'SKMES1', 'NCIH2122', 'VCAP', 'HT29', 'NCIH1650', 'ES2'}

    #sensi_filter = gene_sensitivity['cell_line'].isin(cellline)
    #gene_sensitivity = gene_sensitivity[sensi_filter]
    RF_reg = RandomForestRegressor()
    #X = gene_expression.loc[gene_sensitivity['cell_line'], :].values ## drug ECFP features
    #y = gene_sensitivity.loc[:, 'pIC50'].values ## IC50 values
    # if setting.test_model:
    #     data_length = len(gene_sensitivity)
    #     index_list = [i for i in range(data_length)]
    #     random.shuffle(index_list)
    #     index_lists = [index_list[(data_length//5) * (i): (data_length//5) * (i+1)] for i in range(5)]
    #     for i in range(len(index_lists)):
    #         test_index = index_lists[i]
    #         train_index = list(set(index_list) - set(test_index))
    #         RF_reg.fit(X = X[train_index, :], y = y[train_index])
    #         y_prediction = RF_reg.predict(X[test_index, :])
    #         y_ground = y[test_index]
    #         mse = mean_squared_error(y_ground, y_prediction)
    #         pearson = pearsonr(y_ground, y_prediction)[0]
    #
    #         logger.debug("mse {0}, pearson correlation {1}".format(mse, pearson))
    #         break

    features_importance_df = pd.DataFrame(columns=ECFP.columns)
    for i, cellline in enumerate(set(gene_sensitivity['cell_name'])):

        if i == 5:
            pass
        print(cellline)
        cell_filter = gene_sensitivity['cell_name'] == cellline
        sel_gene_sensi = gene_sensitivity[cell_filter]
        X = ECFP.loc[sel_gene_sensi['drug_name'], :].values
        X_filter = np.isnan(X).all(axis = 1)
        X = X[~X_filter]
        y = sel_gene_sensi.loc[cell_filter, 'IC50'].values
        y = y[~X_filter]
        data_len = len(X)//3


        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=20, stop=100, num=5)] + [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf_random = RandomizedSearchCV(estimator=RF_reg, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=4)

        rf_random.fit(X[data_len:,:], y[data_len:])
        rf_base = RandomForestRegressor(n_estimators = 10, random_state = 42)
        rf_base.fit(X[data_len:, :], y[data_len:])
        y_prediction = rf_base.predict(X[:data_len,:])
        best_random = rf_random.best_estimator_
        best_y_prediction = best_random.predict(X[:data_len,:])
        mse = mean_squared_error(y[:data_len], y_prediction)
        best_mse = mean_squared_error(y[:data_len], best_y_prediction)
        pearson = pearsonr(y[:data_len], y_prediction)[0]
        best_pearson = pearsonr(y[:data_len], best_y_prediction)[0]
        logger.debug("mse: {0!r}, best_mse: {1!r}, pearson: {2!r}, best_pearson: {3!r}".format(mse, best_mse, pearson, best_pearson))
        best_random.fit(X, y)
        features_importance_list = best_random.feature_importances_
        features_importance_dic = {list(ECFP.columns)[j]: features_importance_list[j] for j in range(len(features_importance_list))}
        features_importance_df.loc[cellline] = pd.Series(features_importance_dic)
        
    features_importance_df.to_csv("features_importance_df.csv")
