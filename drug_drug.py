import pandas as pd
import setting
import networkx as nx
import model
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import logging
import os
import pickle

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Recurrent neural network")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def network_propagation(drugs_profile, network):

    # header: ... smiles ...
    # index: genes_name/ENTREZID
    simulated_drug_profile = drugs_profile.copy()
    for entrezid in simulated_drug_profile.index:
        for smile in simulated_drug_profile.columes:
            for neighbor in network.neighbors(entrezid):
                simulated_drug_profile.loc[entrezid, smile] += network[entrezid][neighbor]['relation']

    return simulated_drug_profile

def build_network(network):

    # build a directed network, with node as the entrez id and edge weight as the relationship between two nodes
    G = nx.DiGraph()
    node_1 = network.columns[0]
    node_2 = network.columns[1]
    relation = network.columns[2]
    def add_two_edges(row):
        G.add_edge(row[node_1], row[node_2], relation = row[relation])
        G.add_edge(row[node_2], row[node_1], relation=row[relation])

    network.apply(add_two_edges)
    return G

def split_data(crispr):

    logger.debug("Splitting dataset to training dataset and testing dataset based on genes")
    if os.path.exists(setting.train_index) and os.path.exists(setting.test_index):
        train_index = pickle.load(open(setting.train_index, "rb"))
        test_index = pickle.load(open(setting.test_index, "rb"))
    else:
        train_index, test_index = train_test_split(crispr, n_split = max(len(crispr)/1200, 2), rd_state=0)

        with open(setting.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
        with open(setting.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)

    logger.debug("Splitted data successfully")

    return train_index, test_index

if __name__ == "__main__":

    # Reading synergy score
    # header: Unnamed: 0,drug_a_name, smile_a, drug_b_name, smile_b, cell_line,synergy
    synergy_score_df = pd.read_csv(setting.synergy_score)

    # Read in cell lines gene level dependencies
    # header : index,genes ... cell_lines ... ENTREZID
    # index: genes_name
    cl_genes_dp = pd.read_csv(setting.cl_genes_dp)

    # Read in network data
    # header: ENTREZID(index_level_0) ENTREZID(index_level_1) relation_score
    genes_network = pd.read_csv(setting.genes_network)
    genes_network = build_network(genes_network)

    # Read in drugs REMAP profiles
    # header: ... smiles ...
    # index: genes_name/ENTREZID
    drugs_profile = pd.read_csv(setting.drugs_profile)
    simulated_drugs_profile = network_propagation(drugs_profile, genes_network)

    # Generate final dataset
    drug_a_features = drugs_profile.loc[list(synergy_score_df['smile_a']),:]
    drug_b_features = drugs_profile.loc[list(synergy_score_df['smile_b']),:]
    cl_features = cl_genes_dp[list(synergy_score_df['cell_line'])]
    X = pd.concat([drug_a_features, drug_b_features, cl_features], axis=1)
    Y = synergy_score_df['synergy']

    train_index, test_index = split_data(X.values)

    drug_model = model.DrugsCombModel(drug_a_features = drug_a_features,
                                      drug_b_features = drug_b_features, cl_genes_dp_features=cl_features).get_model()

    training_history = drug_model.fit(x=X.values[train_index], y=Y.values[train_index], validation_split=0.1, epochs=setting.n_epochs,
                                                batch_size=setting.batch_size, verbose=2)

    prediction = drug_model.predict(x=X.values[test_index])
    mse = mean_squared_error(Y.values[test_index], prediction)
    pearson = pearsonr(Y.values[test_index], prediction)

    logging.info("mse: %s, pearson: %s" % (str(mse), str(pearson)))