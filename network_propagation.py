from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import setting
import os
import logging

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def target_as_1_network_propagation(network, drug_target, genes):

    # input drug_target matrix: index = genes, columns = drugs

    # Set target gene feature for a drug to be 1 and non-target gene to be max(probabilites of all edges)

    # if matrix renewal is needed, it will recompute the simulated result matrix
    if not setting.renew and os.path.exists(setting.simulated_result_matrix):

        result_matrix = pd.read_csv(setting.simulated_result_matrix, index_col = 0)

    else:

        # constructed a zero network matrix, columns and index are genes
        network_matrix = get_matrix_from_network(network, genes['entrez'])
        network_matrix.columns = genes['symbol']
        network_matrix.index = genes['symbol']

        # drug_target_matrix: columns = genes, index = Drugs
        drug_target_matrix = drug_target.loc[genes['symbol'], :].T

        # Set target gene feature for a drug to be 1 and non-target gene to be max(probabilites of all edges)
        result_matrix = get_max_probability(drug_target_matrix, network_matrix)
        result_matrix.to_csv(setting.simulated_result_matrix)

    return result_matrix


def get_max_probability(drug_target, network):

    # input drug_target matrix: columns = genes, index = drugs
    result_drug_target = pd.DataFrame(np.zeros(shape=drug_target.shape), columns=drug_target.columns, index = drug_target.index)

    for gene in result_drug_target.columns:
        for drug in result_drug_target.index:
            if drug_target.loc[drug, gene] == 1:
                result_drug_target.loc[drug, gene] = 1
            else:
                result_drug_target.loc[drug, gene] = (drug_target.loc[drug, :] * network.loc[gene, :]).max()

    return result_drug_target

def normalize_matrix(raw_matrix, axis):

    def __normalize(vector):

        denominator = sum(vector)
        if not denominator:
            return vector

        return vector/denominator

    # normalize each row
    normalized_matrix = pd.DataFrame(np.zeros(shape=raw_matrix.shape, dtype='float'), columns=raw_matrix.columns, index=raw_matrix.index)
    if axis == 0 or axis == 'index':

        for i in raw_matrix.index:
            normalized_matrix.loc[i, :] = __normalize(raw_matrix.loc[i, :])

    elif axis == 1 or axis == 'column':

        for i in raw_matrix.columns:
            normalized_matrix.loc[:, i] = __normalize(raw_matrix.loc[:, i])

    else:
        logger.warn("axis is out of range")

    return normalized_matrix

def get_matrix_from_network(network, genes):

    # build the matrix from gene gene interaction network, so far
    # gene-gene self interaction weight is 0
    # output: network_matrix: columns: genes entrezids, index: genes entrezids
    network_matrix = np.zeros(shape=(len(genes), len(genes)), dtype='float')
    network_matrix = pd.DataFrame(data=network_matrix)
    network_matrix.columns, network_matrix.index = genes, genes
    entrez_set = set(genes)

    for row in network.iterrows():

        a, b = int(row[1]['entrez_a']), int(row[1]['entrez_b'])
        if a in entrez_set and b in entrez_set:
            network_matrix.loc[a, b] = row[1]['association']
            network_matrix.loc[b, a] = row[1]['association']

    return network_matrix

def RWlike_network_propagation(network, drug_target, genes):

    # input drug_target matrix: index = genes, columns = drugs
    if not setting.renew and os.path.exists(setting.simulated_result_matrix):

        result_matrix = pd.read_csv(setting.simulated_result_matrix, index_col = 0)

    else:

        # build the matrix from gene gene interaction network, so far
        # gene-gene self interaction weight is 0
        network_matrix = get_matrix_from_network(network, genes['entrez'])

        # Normalize gene gene association probability so that the total gene gene
        # association probability weights for one gene is 1
        normalized_network_matrix = normalize_matrix(network_matrix, 0)
        network_sparse_matrix = csr_matrix(normalized_network_matrix.values)

        # drug_target_matrix: columns = genes, index = Drugs
        drug_target_matrix = drug_target.loc[genes['symbol'], :].values.T

        # result_matrix: columns = genes, index = Drugs
        result_matrix = (csr_matrix(drug_target_matrix).dot(network_sparse_matrix)).todense()

        # Set target genes effect to be 1, because it is 1 in drug_target_matrix and all non-target genes
        # effect is less than 1 since the sum of all weights for a gene is 1
        result_matrix = np.array([result_matrix, drug_target_matrix]).max(axis = 0)
        result_matrix = pd.DataFrame(result_matrix, columns=genes['symbol'], index=drug_target.columns)
        result_matrix.to_csv(setting.simulated_result_matrix)

    return result_matrix