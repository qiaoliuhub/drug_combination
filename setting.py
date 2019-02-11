
activation_method =["relu"]
dropout = 0.3
start_lr = 0.05
lr_decay = 0.001
model_type = 'mlp'
FC_layout = [1024] * 1 + [128] * 1
n_epochs = 200
batch_size = 256

logfile = "../drug_drug/logfile"

synergy_score = "../drug_drug/synergy_score/combin_data_2.csv"
cl_genes_dp = "../drug_drug/cl_gene_dp/complete_cl_gene_dp.csv"
genes_network = "../genes_network/genes_network.csv"
drugs_profile = "../drugs_profile/drugs_profile.csv"

network = "../drug_drug/network/all_tissues_top" # string_network
train_index = "../drug_drug/train_index"
test_index = "../drug_drug/test_index"

renew = True
simulated_result_matrix = "../drug_drug/chemicals/normalized_simulated_result_matrix_string.csv"

ml_train = True
test_ml_train = True
