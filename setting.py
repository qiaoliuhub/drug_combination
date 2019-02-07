
activation_method =["relu", "sigmoid"]
dropout = 0.3
start_lr = 0.001
lr_decay = 0.001
model_type = 'mlp'
FC_layout = [64] * 2 + [32] * 2
n_epochs = 2
batch_size = 128

logfile = "../drug_drug/logfile"

synergy_score = "../drug_drug/synergy_score/combin_data_2.csv"
cl_genes_dp = "../drug_drug/cl_gene_dp/complete_cl_gene_dp.csv"
genes_network = "../genes_network/genes_network.csv"
drugs_profile = "../drugs_profile/drugs_profile.csv"

train_index = "../drug_drug/train_index"
test_index = "../drug_drug/test_index"

renew = True
simulated_result_matrix = "../drug_drug/chemicals/simulated_result_matrix_string.csv"
