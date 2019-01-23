
activation_method =["relu", "sigmoid"]
dropout = 0.3
start_lr = 0.001
lr_decay = 0.001
model_type = 'mlp'
FC_layout = [64] * 2 + [32] * 2
n_epochs = 100
batch_size = 128


synergy_score = "/Users/QiaoLiu1/drug_combin/synergy_score/combin_data_2.csv"
cl_genes_dp = "/Users/QiaoLiu1/drug_combin/cl_gene_dp/complete_cl_gene_dp.csv"
genes_network = "/Users/QiaoLiu1/drug_combin/genes_network/genes_network.csv"
drugs_profile = "/Users/QiaoLiu1/drug_combin/drugs_profile/drugs_profile.csv"

run_specific_log = "/Users/QiaoLiu1/drug_combin/work_dir/logfile"

train_index = "/Users/QiaoLiu1/drug_combin/work_dir/train_index"
test_index = "/Users/QiaoLiu1/drug_combin/work_dir/test_index"
