#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import network_propagation
import model
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import setting
import os
import drug_drug
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
import my_data
from time import time

# setting up nvidia GPU environment
if not setting.ml_train:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def check_data_frames():

    my_data.DrugTargetProfileDataLoader.check_unfound_genes_in_drug_target()
    my_data.GeneDependenciesDataReader.check_unfound_genes_in_gene_dp()
    my_data.DrugTargetProfileDataLoader.check_drugs_in_drug_target()
    my_data.NetworkDataReader.check_genes_in_network()
    my_data.GeneDependenciesDataReader.check_celllines_in_gene_dp()

    ### select only the drugs with features
    ### select only the drug targets in genes

def create_drugs_profiles(raw_chemicals, genes):

    #drug_profile.columns = genes['symbol']
    # columns: drugs, index: genes
    if not setting.drug_profiles_renew and os.path.exists(setting.drug_profiles):
        drug_profile = pd.read_csv(setting.drug_profiles, index_col=0)
        return drug_profile

    drug_profile = np.zeros(shape=(len(raw_chemicals), len(genes)))
    drug_profile = pd.DataFrame(drug_profile, columns=genes['entrez'], index=raw_chemicals['Name'])
    entrez_set = set(genes['entrez'])
    for row in raw_chemicals.iterrows():

        if not isinstance(row[1]['combin_entrez'], str):
            continue

        chem_name, target_list = row[1]['Name'], row[1]['combin_entrez'].split(",")
        for target in target_list:
            target = int(target)
            if target in entrez_set:

                drug_profile.loc[chem_name, target] = 1
    print(setting.drug_profiles)
    drug_profile.T.to_csv(setting.drug_profiles)
    return drug_profile.T



if __name__ == "__main__":

    entrez_set = my_data.GenesDataReader.get_gene_entrez_set()

    ### Reading network data
    ### entrez_a entrez_b association
    ### 1001 10001 0.3
    ### 10001 100001 0.2
    network = my_data.NetworkDataReader.get_network()

    ### Creating test drug target matrix ###
    ###         5-FU  ABT-888  AZD1775  BEZ-235  BORTEZOMIB  CARBOPLATIN
    ### ADH1B      1        0        0        0           0            0
    ### ADRA1A     0        0        0        0           0            0
    ### JAG1       0        1        1        0           1            0
    ### AGTR2      1        1        1        1           0            1
    ### ALB        0        0        0        0           1            0
    drug_target = my_data.DrugTargetProfileDataLoader.get_drug_target_profiles()
    simulated_drug_target = my_data.DrugTargetProfileDataLoader.get_filtered_simulated_drug_target_matrix()
    print(simulated_drug_target, simulated_drug_target.shape)

    ### Reading synergy score data ###
    ### Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
    ### 5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
    ### 5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
    synergy_score = my_data.SynergyDataReader.get_synergy_score()
    print("synergy_score filtered data amount %s" %str(len(synergy_score)))

    exp_drugs = my_data.SynergyDataReader.get_synergy_data_drugs()

    ### Processing gene dependencies map
    ###     "X127399","X1321N1","X143B",
    ### entrez
    ### 1001
    ### 10001
    sel_dp = my_data.GeneDependenciesDataReader.get_gene_dp()

    ### Prepare gene expression data information
    expression_df = my_data.ExpressionDataLoader.prepare_expresstion_df(entrezIDs=list(sel_dp.index),
                                                                        celllines=list(sel_dp.columns))

    ### Ignore drug target genes which have low variance and keep all genes dependencies df genes
    print(simulated_drug_target.shape, sel_dp.shape)
    print(sel_dp)
    print(sel_dp.shape)

    ### Check all data frames schema and contents
    check_data_frames()

    # Generate final dataset
    drug_a_features = simulated_drug_target.loc[list(synergy_score['drug_a_name']), :].values
    drug_b_features = simulated_drug_target.loc[list(synergy_score['drug_b_name']), :].values
    dp_features = sel_dp[list(synergy_score['cell_line'])].T.values
    gene_expression_features = network_propagation.gene_expression_network_propagation(network, expression_df,
                                                                                       entrez_set, drug_target,
                                                                                       synergy_score,
                                                                                       setting.gene_expression_simulated_result_matrix).values
    cl_features_list = []
    if setting.add_dp_feature:
        cl_features_list.append(dp_features)
    if setting.add_ge_feature:
        cl_features_list.append(gene_expression_features)
    cl_features = np.concatenate(tuple(cl_features_list), axis=1)
    X_for = np.concatenate((drug_a_features, drug_b_features, cl_features), axis = 1)
    X_rev = np.concatenate((drug_b_features, drug_a_features, cl_features), axis = 1)
    X = np.concatenate((X_for, X_rev), axis=0)
    scaler = MinMaxScaler()
    #Y = scaler.fit_transform(synergy_score.loc[:, 'synergy'].values.reshape(-1,1)).reshape((-1,))
    Y_half = synergy_score.loc[:, 'synergy'].values.reshape((-1,))
    Y = np.concatenate((Y_half, Y_half), axis=0)
    synergy_score['group'] = synergy_score['drug_a_name'] + '_' + synergy_score['drug_b_name']

    #train_index, test_index = drug_drug.split_data(X_for, group_df=synergy_score, group_col=['group'])
    if setting.index_in_literature:
        synergy_score.reset_index(inplace=True)
        train_index = np.array(synergy_score[synergy_score['fold'] != 0].index)
        test_index = np.array(synergy_score[synergy_score['fold'] == 0].index)
    else:
        train_index, test_index = drug_drug.split_data(X_for, group_df=synergy_score, group_col=['group'])
    train_index = np.concatenate([train_index + X_for.shape[0], train_index])
    test_index_2 = test_index + X_for.shape[0]

    if setting.ml_train:

        x_cols = [x + "_a" for x in list(simulated_drug_target.columns)] + [x + "_b" for x in list(simulated_drug_target.columns)] + list(sel_dp.index)
        X = pd.DataFrame(X, columns=x_cols)
        Y = pd.DataFrame(Y, columns=['synergy'])
        drug_drug.__ml_train(X, Y, train_index, test_index)

    else:
        drug_model = model.DrugsCombModel(drug_a_features = drug_a_features,
                                      drug_b_features = drug_b_features, cl_genes_dp_features=cl_features).get_model()
        logger.info("model information: \n %s" % drug_model.summary())
        logger.debug("Start training")
        tensorboard = TensorBoard(log_dir=setting.tensorboard_log)
        training_history = drug_model.fit(x=X[train_index], y=Y[train_index],
                                                    validation_split=0.1,
                                                    epochs=setting.n_epochs,
                                                    callbacks = [tensorboard],
                                                    verbose=2)


        logger.debug("Training is done")
        train_prediction = drug_model.predict(x=X[train_index]).reshape((-1,))
        train_mse = mean_squared_error(Y[train_index], train_prediction)
        train_pearson = pearsonr(Y[train_index], train_prediction)[0]

        logger.info("training dataset: mse: %s, pearson: %s" % (str(train_mse), str(1-train_pearson**2)))

        test_prediction = drug_model.predict(x=X[test_index]).reshape((-1,))
        test_prediction_2 = drug_model.predict(x=X[test_index_2]).reshape((-1,))
        final_prediction = np.mean([test_prediction, test_prediction_2], axis=0)
        comparison = pd.DataFrame({'ground_truth':Y[test_index],'prediction':final_prediction})
        comparison.to_csv("last_output_{!r}".format(int(time())) + ".csv")
        test_mse = mean_squared_error(Y[test_index], final_prediction)
        test_pearson = pearsonr(Y[test_index], final_prediction)[0]

        logger.info("Testing dataset: mse: %s, pearson: %s" % (str(test_mse), str(1-test_pearson**2)))
