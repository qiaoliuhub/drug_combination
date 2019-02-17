import os
from time import time

# propagation_methods: target_as_1, RWlike
propagation_method = 'RWlike'

activation_method =["relu"]
dropout = [0.2, 0.5, 0.5]
start_lr = 0.03
lr_decay = 0.002
model_type = 'mlp'
FC_layout = [256] * 1 + [64] * 1
n_epochs = 5
batch_size = 256
loss = 'pearson_correlation'
logfile = "../drug_drug/logfile"

synergy_score = "../drug_drug/synergy_score/combin_data_2.csv"
cl_genes_dp = "../drug_drug/cl_gene_dp/complete_cl_gene_dp.csv"
genes_network = "../genes_network/genes_network.csv"
drugs_profile = "../drugs_profile/drugs_profile.csv"

# networks: string_network, all_tissues_top
network = "../drug_drug/network/all_tissues_top"
split_random_seed = 3
index_renewal = True
train_index = "../drug_drug/train_index_" + str(split_random_seed)
test_index = "../drug_drug/test_index_" + str(split_random_seed)

renew = False
simulated_result_matrix = "../drug_drug/chemicals/normalized_simulated_result_matrix_string.csv"

ml_train = False
test_ml_train = True

# estimators: RandomForest, GradientBoosting
estimator = "RandomForest"

if not os.path.exists("../drug_drug/tensorboard_logs"):
    os.mkdir("../drug_drug/tensorboard_logs")
tensorboard_log = "../drug_drug/tensorboard_logs/{}".format(time())
