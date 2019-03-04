#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import network_propagation
import setting
import os
import drug_drug
import my_data
from time import time
import random_test
import torch
from torch import cuda, device
from torch import save, load
from torch.utils import data
import attention_model
import torch.nn.functional as F
import torchsummary
from scipy.stats import pearsonr

# CUDA for PyTorch
use_cuda = cuda.is_available()
if use_cuda:
    device2 = device("cuda:0")
    cuda.set_device(device2)
    cuda.empty_cache()
else:
    device2 = device("cpu")

torch.set_default_tensor_type('torch.FloatTensor')
# cudnn.benchmark = True

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

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
    drug_target = random_test.create_drugs_profiles(raw_chemicals, genes)

    ### Get simulated drug_target
    ### columns=genes['symbol'], index=drugs
    raw_simulated_drug_target = random_test.simulated_drug_target_matrix(network, drug_target, genes)
    simulated_drug_target = raw_simulated_drug_target.loc[~(raw_simulated_drug_target == 0).all(axis = 1), :]
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
    random_test.check_data_frames(drug_target, sel_dp, network, genes, cell_lines, exp_drugs)

    ### Ignore genes that is in genes dependencies and not in genes
    merged_sel_dp = sel_dp.merge(genes, left_index=True, right_on='entrez')
    sel_dp = merged_sel_dp.set_index('entrez').drop(['symbol'], axis = 1)

    ### Ignore drug target genes which have low variance and keep all genes dependencies df genes
    print(simulated_drug_target.shape, sel_dp.shape)
    gene_filter = (simulated_drug_target.var(axis=0) > 0)
    sel_drug_target = simulated_drug_target.loc[:, gene_filter]
    sel_dp_filter = (sel_dp.var(axis=1) > 0)
    sel_dp = sel_dp.loc[sel_dp_filter, :]
    print(sel_drug_target)
    print(sel_drug_target.shape, sel_dp.shape)

    ### Prepare gene expression data information
    expression_data_loader = my_data.ExpressionDataLoader()
    expression_df = expression_data_loader.prepare_expresstion_df(entrezIDs=list(merged_sel_dp.entrez), celllines=list(sel_dp.columns))
    logger.debug("raw expression is ready")

    # Generate final dataset
    drug_a_features = sel_drug_target.loc[list(synergy_score['drug_a_name']), :]
    drug_b_features = sel_drug_target.loc[list(synergy_score['drug_b_name']), :]
    dp_features = sel_dp[list(synergy_score['cell_line'])].T
    gene_expression_features = network_propagation.gene_expression_network_propagation(network, expression_df,
                                                                                       genes, drug_target,
                                                                                       synergy_score,
                                                                                       setting.gene_expression_simulated_result_matrix)
    logger.debug("preprocessed expression is ready")

    final_index_1 = synergy_score.apply(lambda row: row['drug_a_name']+'_'+row['drug_b_name']+'_' +row['cell_line'], axis = 1)
    final_index_2 = synergy_score.apply(lambda row: row['drug_b_name']+'_'+row['drug_a_name']+'_' +row['cell_line'], axis = 1)
    half_df_1 = pd.concat([drug_a_features, drug_b_features, dp_features, gene_expression_features], axis=0)
    half_df_2 = pd.concat([drug_b_features, drug_a_features, dp_features, gene_expression_features], axis=0)
    half_df_1.fillna(0, inplace = True)
    half_df_2.fillna(0, inplace = True)
    logger.debug("Reshaping the ndarrray")
    half_df_1 = half_df_1.values.reshape((4, -1, half_df_1.shape[-1])).transpose(1,0,2)
    half_df_2 = half_df_2.values.reshape((4, -1, half_df_2.shape[-1])).transpose(1,0,2)

    Y_labels = synergy_score.loc[:, 'synergy']
    Y_half = Y_labels.values.reshape((-1,))
    Y = np.concatenate((Y_half, Y_half), axis=0)
    synergy_score['group'] = synergy_score['drug_a_name'] + '_' + synergy_score['drug_b_name']

    train_index, test_index = drug_drug.split_data(half_df_1, group_df=synergy_score, group_col=['fold'])
    #train_index = np.concatenate([train_index + half_df_1.shape[0], train_index])
    test_index_2 = test_index + half_df_1.shape[0]
    train_index, test_index, test_index_2 = train_index[:100], test_index[:100], test_index_2[:100]

    for i, combin_drug_feature_array in enumerate(half_df_1[train_index,]):
        if i<=101:
            save(combin_drug_feature_array, os.path.join('datas', str(final_index_1.iloc[train_index[i]]) + '.pt'))
    for i, combin_drug_feature_array in enumerate(half_df_1[test_index,]):
        if i<=101:
            save(combin_drug_feature_array, os.path.join('datas', str(final_index_1.iloc[test_index[i]]) + '.pt'))

    partition = {'train': list(final_index_1.iloc[train_index]),
                 'test1': list(final_index_1.iloc[test_index]), 'test2': list(final_index_2.iloc[test_index])}

    #+ list(final_index_2.loc[train_index])
    labels = {key: value for key, value in zip(list(final_index_1) + list(final_index_2),
                                               list(Y_labels) * 2)}

    train_params = {'batch_size': 64,
              'shuffle': True}
    test_params = {'batch_size': len(test_index),
                   'shuffle': True}

    logger.debug("Preparing datasets ... ")
    training_set = my_data.MyDataset(partition['train'], labels)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = my_data.MyDataset(partition['test1'], labels)
    validation_generator = data.DataLoader(validation_set, **test_params)

    logger.debug("Preparing models")
    drug_model = attention_model.get_model()
    torchsummary.summary(drug_model, input_size=[(setting.n_feature_type, setting.d_model), (setting.n_feature_type, setting.d_model)])
    optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay, betas=(0.9, 0.98), eps=1e-9)

    logger.debug("Start training")
    # Loop over epochs
    for epoch in range(setting.n_epochs):

        drug_model.train()
        start = time()
        cptime = start
        total_loss = 0
        i = 0

        # Training
        for local_batch, local_labels in training_generator:
            i += 1
            # Transfer to GPU
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)

            # Model computations
            preds = drug_model(local_batch, local_batch).view(-1)
            ys = local_labels.contiguous().view(-1)
            optimizer.zero_grad()
            assert preds.size(-1) == ys.size(-1)
            loss = F.mse_loss(preds, ys)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            n_iter = 1
            if (i + 1) % n_iter == 0:
                p = int(100 * (i + 1) / setting.batch_size)
                avg_loss = total_loss / n_iter
                random_test.logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss))
                total_loss = 0

        # Testing
        test_i = 0
        test_total_loss = 0
        test_loss = []

        with torch.set_grad_enabled(False):

            drug_model.eval()
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                test_i += 1
                local_labels_on_cpu = local_labels
                local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)

                # Model computations
                preds = drug_model(local_batch, local_batch).contiguous().view(-1)
                ys = local_labels.contiguous().view(-1)
                assert preds.size(-1) == ys.size(-1)
                loss = F.mse_loss(preds, ys)
                prediction_on_cpu = preds.cpu().numpy()
                pearson_loss = pearsonr(prediction_on_cpu,local_labels_on_cpu)[0]
                test_total_loss += loss.item()

                n_iter = 1
                if (test_i + 1) % n_iter == 0:
                    avg_loss = test_total_loss / n_iter
                    test_loss.append(avg_loss)
                    test_total_loss = 0

        logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r}".format(sum(test_loss)/len(test_loss), pearson_loss))