import pandas as pd
import setting
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import logging

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':

    ECFP = pd.read_csv(setting.ECFP_file)
    ECFP = ECFP[['Name', 'ECFP_6']]
    ECFP.set_index('Name', inplace=True)
    ECFP = ECFP['ECFP_6'].apply(lambda i: pd.Series(list(i))).astype(int)
    ECFP = ECFP.loc[:, ~((ECFP == 0).all(axis=0))]
    gene_expression = pd.read_csv(setting.gene_expression_file).T

    gene_sensitivity = pd.read_csv(setting.gene_sensitivity_file)
    # cellline = {'RKO', 'OCUBM', 'NCIH520', 'UACC62', 'A375', 'T47D', 'MDAMB436', 'KPL1', 'LOVO', 'SKMEL30', 'ZR751', 'A2058',
    #  'OV90', 'SW837', 'HCT116', 'NCIH23', 'SKOV3', 'NCIH460', 'A2780', 'SW620', 'A427', 'DLD1', 'RPMI7951', 'HT144',
    #  'CAOV3', 'SKMES1', 'NCIH2122', 'VCAP', 'HT29', 'NCIH1650', 'ES2'}
    cellline = ''

    sensi_filter = gene_sensitivity['cell_line'].isin(cellline)
    gene_sensitivity = gene_sensitivity[sensi_filter]
    RF_reg = RandomForestRegressor(n_jobs=6, random_state=33,verbose=2)
    X = gene_expression.loc[gene_sensitivity['cell_line'], :].values ## drug ECFP features
    y = gene_sensitivity.loc[:, 'pIC50'].values ## IC50 values
    if setting.test_model:
        data_length = len(gene_sensitivity)
        index_list = [i for i in range(data_length)]
        random.shuffle(index_list)
        index_lists = [index_list[(data_length//5) * (i): (data_length//5) * (i+1)] for i in range(5)]
        for i in range(len(index_lists)):
            test_index = index_lists[i]
            train_index = list(set(index_list) - set(test_index))
            RF_reg.fit(X = X[train_index, :], y = y[train_index])
            y_prediction = RF_reg.predict(X[test_index, :])
            y_ground = y[test_index]
            mse = mean_squared_error(y_ground, y_prediction)
            pearson = pearsonr(y_ground, y_prediction)[0]

            logger.debug("mse {0}, pearson correlation {1}".format(mse, pearson))
            break

    features_importance_df = pd.DataFrame(columns=gene_expression.columns)
    for i, drug in enumerate(set(gene_sensitivity['drug_name'])):

        if i == 5:
            break
        print(drug)
        drug_filter = gene_sensitivity['drug_name'] == drug
        sel_gene_sensi = gene_sensitivity[drug_filter]
        X = gene_expression.loc[sel_gene_sensi['cell_line'], :].values
        y = gene_sensitivity.loc[drug_filter, 'pIC50'].values
        data_len = len(X)//3
        RF_reg.fit(X[data_len:,:], y[data_len:])
        y_prediction = RF_reg.predict(X[:data_len,:])
        mse = mean_squared_error(y[:data_len], y_prediction)
        pearson = pearsonr(y[:data_len], y_prediction)[0]
        print(mse, pearson)
        features_importance_list = RF_reg.feature_importances_
        features_importance_dic = {list(gene_expression.columns)[j]: features_importance_list[j] for j in range(len(features_importance_list))}
        features_importance_df.loc[i] = pd.Series(features_importance_dic)

