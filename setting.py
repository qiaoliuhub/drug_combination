import os
from time import time

unit_test = True

working_dir = '/Users/QiaoLiu1/drug_combin/drug_drug'
# propagation_methods: target_as_1, RWlike, random_walk
propagation_method = 'random_walk'
# feature type: F_representation, others, determine whether or not ignoring drugs without hidden representation
feature_type = 'others'
F_repr_feature_length = 1000

activation_method =["relu"]
dropout = [0.2, 0.1, 0.1]
start_lr = 0.01
lr_decay = 0.001
model_type = 'mlp'
FC_layout = [256] * 1 + [64] * 1
n_epochs = 5
batch_size = 256
loss = 'mse'
logfile = os.path.join(working_dir, 'logfile')
NBS_logfile = os.path.join(working_dir, 'NBS_logfile')

synergy_score = os.path.join(working_dir, 'synergy_score', 'combin_data_2.csv')
cl_genes_dp = os.path.join(working_dir, 'cl_gene_dp', 'complete_cl_gene_dp.csv')
#genes_network = '../genes_network/genes_network.csv'
#drugs_profile = '../drugs_profile/drugs_profile.csv'
F_drug = os.path.join(working_dir, 'F_repr', 'sel_F_drug_sample.csv')
F_cl = os.path.join(working_dir, 'F_repr', 'sel_F_cl_sample.csv')

ECFP = os.path.join(working_dir, 'chemicals', 'ECFP6.csv')
physicochem = os.path.join(working_dir, 'chemicals', 'physicochemical_des.csv')

# networks: string_network, all_tissues_top
network = os.path.join(working_dir, 'network', 'string_network')
network_matrix = os.path.join(working_dir, 'network', 'string_network_matrix.csv')
split_random_seed = 3
index_in_literature = True
index_renewal = True
train_index = os.path.join(working_dir, 'train_index_' + str(split_random_seed))
test_index = os.path.join(working_dir, 'test_index_' + str(split_random_seed))

renew = False
RWlike_simulated_result_matrix = os.path.join(working_dir, 'chemicals', 'normalized_simulated_result_matrix_string.csv')
target_as_1_simulated_result_matrix = os.path.join(working_dir, 'chemicals', 'target_1_simulated_result_matrix_string.csv')
target_as_0_simulated_result_matrix = os.path.join(working_dir, 'chemicals', 'target_0_simulated_result_matrix_string.csv')
gene_expression_simulated_result_matrix = os.path.join(working_dir, 'chemicals', 'gene_expression_simulated_result_matrix_string.csv')
random_walk_simulated_result_matrix = os.path.join(working_dir, 'chemicals', 'random_walk_simulated_result_matrix')
intermediate_ge_target0_matrix = os.path.join(working_dir, 'chemicals', 'intermediate_ge_target0_matrix')

ml_train = False
test_ml_train = True

# estimators: RandomForest, GradientBoosting
estimator = "RandomForest"

if not os.path.exists(os.path.join(working_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(working_dir, 'tensorboard_logs'))
tensorboard_log = os.path.join(working_dir, "tensorboard_logs/{}".format(time()))

combine_gene_expression_renew = False
expression_data_renew = False
gene_expression = "/Users/QiaoLiu1/microbiome/trial/CCLE.tsv"
backup_expression = "/Users/QiaoLiu1/microbiome/trial/GDSC.tsv"
processed_expression = os.path.join(working_dir, 'processed_expression.csv')


combine_drug_target_renew = False
combine_drug_target_matrix = os.path.join(working_dir, 'chemicals', 'combine_drug_target_matrix.csv')

drug_profiles_renew = False
drug_profiles = os.path.join(working_dir, 'chemicals','drug_profiles.csv')


python_interpreter_path = '/Users/QiaoLiu1/anaconda3/envs/pynbs_env/bin/python'

y_transform = True

cellline_features = ['gene_dependence', 'gene_expression', 'cl_F_repr']
drug_features = ['drug_target_profile', 'drug_physiochemistry', 'ECFP', 'drug_F_repr']
update_features = True
output_FF_layers = [400, 1]
d_input = 400
n_feature_type = 2 + len(cellline_features)
if feature_type == 'F_representation':
    n_feature_type = 3
    d_input = 1000
d_model = 200
attention_heads = 8
attention_dropout = 0.2
n_layers = 1 # This has to be 1
