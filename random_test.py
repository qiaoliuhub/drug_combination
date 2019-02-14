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


def check_data_frames(drug_target, sel_dp, network, genes, cell_lines, exp_drugs):

    ### Make sure that drug target genes and gene dependencies genes are in selected ~2300 genes
    merged_drug_target = drug_target.merge(genes, left_index=True, right_on='symbol')
    logger.info("merged_drug_targets: %s" %str(merged_drug_target.head()))
    print(merged_drug_target.shape)
    unfound_genes = set(merged_drug_target.symbol) - set(genes['symbol'])
    if len(unfound_genes) == 0:
        logger.info("Found all genes in drug target")
    else:
        logger.info("Unfound genes in drug target: %s" % str(unfound_genes))

    merged_sel_dp = sel_dp.merge(genes, left_index=True, right_on='entrez')
    print(merged_sel_dp.shape)
    logger.info("merged_sel_dp: %s" % str(merged_sel_dp.head()))

    unfound_genes = set(merged_sel_dp.symbol) - set(genes['symbol'])
    if len(unfound_genes) == 0:
        logger.info("Found all genes in genes dependencies")
    else:
        logger.info("Unfound genes in genes dependencies: %s" % str(unfound_genes))


    ### Make sure drugs are all in drug_target dataframe
    unfound_drugs = exp_drugs - set(drug_target.columns)
    if len(unfound_drugs) == 0:
        logger.info("Found all Drugs")
    else:
        logger.info("Unfound Drugs: %s" % str(unfound_drugs))

    ### Make sure genes are all in network
    unfound_genes = set(genes['entrez']) - (set(network['entrez_a']).union(set(network['entrez_b'])))
    if len(unfound_genes) == 0:
        logger.info("Found all genes in networks")
    else:
        logger.info("Unfound genes in networks: %s" % str(unfound_genes))

    ### Make sure cell lines in synergy score dataframe are in that in dependencies scores
    unfound_cl = set(merged_sel_dp.columns) - cell_lines - set(['symbol', 'entrez'])
    if len(unfound_cl) == 0:
        logger.info("Found all cell lines")
    else:
        logger.info("Unfound cell lines: %s" % str(unfound_cl))

    ### select only the drugs with features
    ### select only the drug targets in genes

def create_drugs_profiles(raw_chemicals, genes):

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

    drug_profile.columns = genes['symbol']

    # columns: drugs, index: genes
    return drug_profile.T

def simulated_drug_target_matrix(network, drug_target, genes):

    if setting.propagation_method == 'target_as_1':
        simulated_drug_target_matrix = network_propagation.target_as_1_network_propagation(network, drug_target, genes)

    elif setting.propagation_method == 'target_as_0':
        simulated_drug_target_matrix = network_propagation.target_as_0_network_propagation(network, drug_target, genes)

    else:
        simulated_drug_target_matrix = network_propagation.RWlike_network_propagation(network, drug_target, genes)

    return simulated_drug_target_matrix

if __name__ == "__main__":

    genes = pd.read_csv("../drug_drug/Genes/combin_genes.csv")
    drugs = pd.read_csv("../drug_drug/chemicals/smiles_merck.csv")

    ### Reading network data
    ### entrez_a entrez_b association
    ### 1001 10001 0.3
    ### 10001 100001 0.2
    raw_network = pd.read_csv(setting.network, header=None, sep = '\t')
    raw_network.columns = ['entrez_a', 'entrez_b', 'association']
    network = raw_network[(raw_network['entrez_a'].isin(genes['entrez'])) & (raw_network['entrez_b'].isin(genes['entrez']))]

    ### Creating test drug target matrix ###
    ###         5-FU  ABT-888  AZD1775  BEZ-235  BORTEZOMIB  CARBOPLATIN
    ### ADH1B      1        0        0        0           0            0
    ### ADRA1A     0        0        0        0           0            0
    ### JAG1       0        1        1        0           1            0
    ### AGTR2      1        1        1        1           0            1
    ### ALB        0        0        0        0           1            0

    # drug_target_dict = {}
    # for drug in drugs['Name']:
    #
    #     drug_target_dict[drug] = pd.Series(np.random.randint(2, size=len(genes)))
    #
    # drug_target = pd.DataFrame(data=drug_target_dict)
    # drug_target.index = genes['symbol']
    raw_chemicals = pd.read_csv("../drug_drug/chemicals/raw_chemicals.csv")
    drug_target = create_drugs_profiles(raw_chemicals, genes)

    ### Get simulated drug_target
    ### columns=genes['symbol'], index=drugs
    raw_simulated_drug_target = simulated_drug_target_matrix(network, drug_target, genes)
    simulated_drug_target = raw_simulated_drug_target.loc[~raw_simulated_drug_target.isnull().all(axis = 1), :]
    sel_drugs = set(simulated_drug_target.index)
    print(simulated_drug_target, simulated_drug_target.shape)

    ### Reading synergy score data ###
    ### Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
    ### 5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
    ### 5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
    synergy_score = pd.read_csv("../drug_drug/synergy_score/combin_data_2.csv")
    synergy_score = synergy_score[(synergy_score['drug_a_name'].isin(sel_drugs)) & (synergy_score['drug_b_name'].isin(sel_drugs))]
    print("synergy_score filtered data amount %s" %str(len(synergy_score)))
    cell_lines = set(synergy_score['cell_line'])
    exp_drugs = set(synergy_score['drug_a_name']).union(set(synergy_score['drug_b_name']))

    ### Processing gene dependencies map
    ###     "X127399","X1321N1","X143B",
    ### entrez
    ### 1001
    ### 10001

    cl_gene_dp_indexes = pd.read_csv("../drug_drug/cl_gene_dp/all_dependencies_gens.csv", usecols = ['symbol', 'entrez'])
    cl_gene_dp = pd.read_csv("../drug_drug/cl_gene_dp/complete_cl_gene_dp_1.csv")
    cl_gene_dp.index = cl_gene_dp_indexes['entrez']
    cl_gene_dp.columns = list(map(lambda x: x.split("_")[0], cl_gene_dp.columns))
    sel_dp = cl_gene_dp[list(cell_lines)].reset_index().drop_duplicates(subset='entrez').set_index('entrez')

    ### Check all data frames schema and contents
    check_data_frames(drug_target, sel_dp, network, genes, cell_lines, exp_drugs)

    ### Ignore genes that is in genes dependencies and not in genes
    merged_sel_dp = sel_dp.merge(genes, left_index=True, right_on='entrez')
    sel_dp = merged_sel_dp.set_index('symbol').drop(['entrez'], axis = 1)

    ### Ignore drug target genes which have low variance and keep all genes dependencies df genes
    gene_filter = (simulated_drug_target.var(axis=0) > 0)
    sel_drug_target = simulated_drug_target.loc[:, gene_filter]
    print(sel_drug_target)

    # Generate final dataset
    drug_a_features = sel_drug_target.loc[list(synergy_score['drug_a_name']), :].values
    drug_b_features = sel_drug_target.loc[list(synergy_score['drug_b_name']), :].values
    cl_features = sel_dp[list(synergy_score['cell_line'])].T.values
    X_for = np.concatenate((drug_a_features, drug_b_features, cl_features), axis = 1)
    X_rev = np.concatenate((drug_b_features, drug_a_features, cl_features), axis = 1)
    X = np.concatenate((X_for, X_rev), axis=0)
    scaler = MinMaxScaler()
    #Y = scaler.fit_transform(synergy_score.loc[:, 'synergy'].values.reshape(-1,1)).reshape((-1,))
    Y_half = synergy_score.loc[:, 'synergy'].values.reshape((-1,))
    Y = np.concatenate((Y_half, Y_half), axis=0)

    train_index, test_index = drug_drug.split_data(X_for)
    train_index = np.concatenate([train_index + X_for.shape[0], train_index])
    test_index = np.concatenate([test_index + X_for.shape[0], test_index])

    if setting.ml_train:

        x_cols = [x + "_a" for x in list(sel_drug_target.columns)] + [x + "_b" for x in list(sel_drug_target.columns)] + list(sel_dp.index)
        X = pd.DataFrame(X, columns=x_cols)
        Y = pd.DataFrame(Y, columns=['synergy'])
        drug_drug.__ml_train(X, Y, train_index, test_index)

    else:
        drug_model = model.DrugsCombModel(drug_a_features = drug_a_features,
                                      drug_b_features = drug_b_features, cl_genes_dp_features=cl_features).get_model()
        logger.info("model information: \n %s" % drug_model.summary())
        logger.debug("Start training")
        training_history = drug_model.fit(x=X[train_index], y=Y[train_index], validation_split=0.1, epochs=setting.n_epochs,
                                                    verbose=2)

        logger.debug("Training is done")
        train_prediction = drug_model.predict(x=X[train_index]).reshape((-1,))
        train_mse = mean_squared_error(Y[train_index], train_prediction)
        train_pearson = pearsonr(Y[train_index], train_prediction)

        logger.info("training dataset: mse: %s, pearson: %s" % (str(train_mse), str(train_pearson)))

        test_prediction = drug_model.predict(x=X[test_index]).reshape((-1,))
        test_mse = mean_squared_error(Y[test_index], test_prediction)
        test_pearson = pearsonr(Y[test_index], test_prediction)

        logger.info("Evaluation dataset: mse: %s, pearson: %s" % (str(test_mse), str(test_pearson)))
