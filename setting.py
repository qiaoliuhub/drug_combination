# propagation_methods: target_as_1, RWlike
propagation_method = 'target_as_1'

activation_method =["relu"]
dropout = 0.3
start_lr = 0.05
lr_decay = 0.001
model_type = 'mlp'
FC_layout = [256] * 1 + [64] * 1
n_epochs = 200
batch_size = 256

logfile = "../drug_drug/logfile"

synergy_score = "../drug_drug/synergy_score/combin_data_2.csv"
cl_genes_dp = "../drug_drug/cl_gene_dp/complete_cl_gene_dp.csv"
genes_network = "../genes_network/genes_network.csv"
drugs_profile = "../drugs_profile/drugs_profile.csv"

# networks: string_network, all_tissues_top
network = "../drug_drug/network/all_tissues_top"
split_random_seed = 3
train_index = "../drug_drug/train_index_" + str(split_random_seed)
test_index = "../drug_drug/test_index_" + str(split_random_seed)

renew = True
simulated_result_matrix = "../drug_drug/chemicals/target_simulated_result_matrix_string.csv"

ml_train = False
test_ml_train = True
estimator = "RandomForest"

