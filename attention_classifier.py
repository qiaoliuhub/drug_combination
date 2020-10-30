#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
import setting
from os import path, mkdir, environ
import my_data
from time import time
import random_test
import torch
from torch import cuda, device
from torch import save, load
from torch.utils import data
import attention_model
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score
from imblearn.over_sampling import RandomOverSampler
import random
import shap
import drug_drug
import pickle
import wandb
import sys
sys.path.append(path.dirname(path.realpath(__file__)) + '/NeuralFingerPrint')
import data_utils
import timeit
import pdb

USE_wandb = True
if USE_wandb:
    pdb.set_trace()
    wandb.init(project="Drug combination hyper",
               name = setting.run_dir + '_' + setting.data_specific[:30],
               notes = setting.data_specific)
else:
    environ["WANDB_MODE"] = "dryrun"

# CUDA for PyTorch
use_cuda = cuda.is_available()
if use_cuda:
    device2 = device("cuda:0")
    cuda.set_device(device2)
    cuda.empty_cache()
else:
    device2 = device("cpu")

torch.set_default_tensor_type('torch.FloatTensor')

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped

def get_final_index():
    ## get the index of synergy score database
    if not setting.update_final_index and path.exists(setting.final_index):
        final_index = pd.read_csv(setting.final_index, header=None)[0]
    else:
        final_index = my_data.SynergyDataReader.get_final_index()
    return final_index


def prepare_data():
    ## avoid repeating calculate input and output data
    if not setting.update_xy and path.exists(setting.old_x) and path.exists(setting.old_y):
        X = np.load(setting.old_x)
        with open(setting.old_x_lengths, 'rb') as old_x_lengths:
            drug_features_length, cellline_features_length = pickle.load(old_x_lengths)
        with open(setting.old_y, 'rb') as old_y:
            Y = pickle.load(old_y)
    else:
        X, drug_features_length, cellline_features_length = \
            my_data.SamplesDataLoader.Raw_X_features_prep(methods='flexible_attn')
        np.save(setting.old_x, X)
        with open(setting.old_x_lengths, 'wb+') as old_x_lengths:
            pickle.dump((drug_features_length, cellline_features_length), old_x_lengths)

        Y = my_data.SamplesDataLoader.Y_features_prep()
        with open(setting.old_y, 'wb+') as old_y:
            pickle.dump(Y, old_y)
    return X, Y, drug_features_length, cellline_features_length

def prepare_model(reorder_tensor, entrez_set):

    ### prepare two models
    ### drug_model: the one used for training
    ### best_drug_mode;: the one used for same the best model

    mask = drug_drug.transfer_df_to_mask(torch.load(setting.pathway_dataset), entrez_set).T
    #final_mask = pd.concat([mask for _ in range(setting.d_model_i)], axis=1).values
    final_mask = None
    drug_model = attention_model.get_multi_models(reorder_tensor.get_reordered_slice_indices(), input_masks=final_mask,
                                                  drugs_on_the_side = True, classifier = False)
    best_drug_model = attention_model.get_multi_models(reorder_tensor.get_reordered_slice_indices(), input_masks=final_mask,
                                                       drugs_on_the_side=True, classifier=False)
    for n, m in drug_model.named_modules():
        if n == "out":
            m.register_forward_hook(drug_drug.input_hook)
    for best_n, best_m in best_drug_model.named_modules():
        if best_n == "out":
            best_m.register_forward_hook(drug_drug.input_hook)
    drug_model = drug_model.to(device2)
    best_drug_model = best_drug_model.to(device2)
    if USE_wandb:
        wandb.watch(drug_model, log="all")
    return drug_model, best_drug_model

def persist_data_as_data_point_file(local_X, final_index_for_X):

    ### prepare files for dataloader
    for i, combin_drug_feature_array in enumerate(local_X):
        if setting.unit_test:
            if i <= 501:  # and not path.exists(path.join('datas', str(final_index_for_X.iloc[i]) + '.pt')):
                save(combin_drug_feature_array, path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt'))
        else:
            if setting.update_features or not path.exists(
                    path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt')):
                save(combin_drug_feature_array, path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt'))

def prepare_splitted_dataset(partition, labels):

    ### prepare train, test, evaluation data generator

    logger.debug("Preparing datasets ... ")
    training_set = my_data.MyDataset(partition['train'] + partition['eval1'] + partition['eval2'], labels)
    train_params = {'batch_size': setting.batch_size,
                    'shuffle': True}
    training_generator = data.DataLoader(training_set, **train_params)

    eval_train_set = my_data.MyDataset(partition['train'] + partition['eval1'] + partition['eval2'], labels)
    training_index_list = partition['train'] + partition['eval1'] + partition['eval2']
    logger.debug("Training data length: {!r}".format(len(training_index_list)))
    eval_train_params = {'batch_size': setting.batch_size,
                         'shuffle': False}
    eval_train_params1 = {'batch_size': len(partition['train']) // 2,
                          'shuffle': False}

    ## used for SHAP analysis
    eval_train_shap_generator = data.DataLoader(eval_train_set, **eval_train_params1)
    eval_train_generator = data.DataLoader(eval_train_set, **eval_train_params)

    # validation_set = my_data.MyDataset(partition['eval1'] + partition['eval2'], labels)
    validation_set = my_data.MyDataset(partition['test1'], labels)
    eval_params = {'batch_size': len(partition['test1']),
                   'shuffle': False}
    validation_generator = data.DataLoader(validation_set, **eval_params)

    test_set = my_data.MyDataset(partition['test1'], labels)
    test_index_list = partition['test1']  # + partition['test2']
    logger.debug("Test data length: {!r}".format(len(test_index_list)))
    pickle.dump(test_index_list, open("test_index_list", "wb+"))
    test_params = {'batch_size': len(test_index_list) // 4,
                   'shuffle': False}
    test_generator = data.DataLoader(test_set, **test_params)

    return training_generator, eval_train_generator, eval_train_shap_generator, validation_generator, test_generator

def run():


    final_index = get_final_index()
    ## get genes
    entrez_set = my_data.GenesDataReader.get_gene_entrez_set()

    logger.debug("Getting features and synergy scores ...")
    X, Y, drug_features_length, cellline_features_length = prepare_data()

    ## Construct a classification problem, if synergy score is larger than 30, we set it as positive samples
    assert setting.output_FF_layers == 2, "classification model should have output dim as 2"
    Y = (Y>=30).astype(int)

    logger.debug("Preparing models")
    slice_indices = drug_features_length + drug_features_length + cellline_features_length
    reorder_tensor = drug_drug.reorganize_tensor(slice_indices, setting.arrangement, 2)
    logger.debug("the layout of all features is {!r}".format(reorder_tensor.get_reordered_slice_indices()))
    drug_model, best_drug_model = prepare_model(reorder_tensor, entrez_set)

    optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min = 1e-7)

    test_generator = None
    eval_train_shap_generator = None
    test_index_list = None
    test_params = None
    best_auc = 0

    split_func = my_data.DataPreprocessor.reg_train_eval_test_split

    logger.debug("Spliting data ...")
    for train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 in split_func(fold='drug_fold'):

        local_X = X[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]
        final_index_for_X = final_index.iloc[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]
        ori_Y = Y
        persist_data_as_data_point_file(local_X, final_index_for_X)

        #### Deal with imbalanced data problem with negative sampling
        ros = RandomOverSampler(random_state=42)
        resample_train_index = np.concatenate((train_index, evaluation_index, evaluation_index_2))
        _ = ros.fit_resample(X[resample_train_index],
                             Y[resample_train_index])
        new_train_index = resample_train_index[ros.sample_indices_]
        oversample_train_index = list(new_train_index)
        random.shuffle(oversample_train_index)

        partition = {'train': list(final_index.iloc[train_index]),
                     'test1': list(final_index.iloc[test_index]), 'test2': list(final_index.iloc[test_index_2]),
                     'eval1': list(final_index.iloc[evaluation_index]),
                     'eval2': list(final_index.iloc[evaluation_index_2])}

        assert len(set(oversample_train_index) & set(test_index)) == 0
        assert len(set(oversample_train_index) & set(resample_train_index)) == len(set(resample_train_index))

        labels = {key: value for key, value in zip(list(final_index),
                                                   list(Y.reshape(-1)))}
        ori_labels = {key: value for key, value in zip(list(final_index),
                                                   list(ori_Y.reshape(-1)))}
        save(ori_labels, setting.y_labels_file)

        training_generator, eval_train_generator, eval_train_shap_generator, \
        validation_generator, test_generator = prepare_splitted_dataset(partition, labels)
        training_index_list = partition['train'] + partition['eval1'] + partition['eval2']
        test_index_list = partition['test1']

        logger.debug("Start training")

        for epoch in range(setting.n_epochs):

            drug_model.train()
            start = time()
            cur_epoch_train_loss = []
            train_total_loss = 0
            train_i = 0
            all_preds = []
            all_ys = []
            # Training
            for (local_batch, smiles_a, smiles_b), local_labels in training_generator:
                train_i += 1
                # Transfer to GPU
                local_labels_on_cpu = np.array(local_labels).reshape(-1)
                sample_size = local_labels_on_cpu.shape[0]
                local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
                local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
                reorder_tensor.load_raw_tensor(local_batch)
                local_batch = reorder_tensor.get_reordered_narrow_tensor()
                # drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
                # drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
                # drugs = (drug_a, drug_b)
                # preds = drug_model(*local_batch, drugs = drugs)
                # Model computations
                preds = drug_model(*local_batch)
                preds = preds.contiguous().view(-1)
                optimizer.zero_grad()
                ys = local_labels.contiguous().view(-1)
                assert preds.size(0) == ys.size(0)
                # loss = F.nll_loss(preds, ys)
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(preds, ys.float())
                loss.backward()
                optimizer.step()
                prediction_on_cpu = preds.detach().cpu().numpy()
                # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                #                                   prediction_on_cpu[sample_size:]], axis=0)
                mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                all_preds.append(mean_prediction_on_cpu)
                all_ys.append(local_labels_on_cpu)

                train_total_loss += loss.item()

                n_iter = 50
                if train_i % n_iter == 0:
                    sample_size = len(train_index) + 2 * len(evaluation_index)
                    p = int(100 * train_i * setting.batch_size/sample_size)
                    avg_loss = train_total_loss / n_iter
                    random_test.logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                          ((time() - start) // 60, epoch, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, avg_loss))
                    train_total_loss = 0
                    cur_epoch_train_loss.append(avg_loss)

            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys).astype(int)
            assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
            val_train_roc_auc = roc_auc_score(all_ys.reshape(-1), all_preds.reshape(-1))
            val_train_pr_auc = average_precision_score(all_ys.reshape(-1), all_preds.reshape(-1))


            ### Evaluation
            # val_train_i = 0
            # save_data_num = 0
            #
            with torch.set_grad_enabled(False):

                drug_model.eval()
            #     all_preds = []
            #     all_ys = []
            #     for (local_batch, smiles_a, smiles_b), local_labels in eval_train_generator:
            #         val_train_i += 1
            #         local_labels_on_cpu = np.array(local_labels).reshape(-1)
            #         sample_size = local_labels_on_cpu.shape[0]
            #         local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            #         # Transfer to GPU
            #         local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
            #         reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length))
            #         local_batch = reorder_tensor.get_reordered_narrow_tensor()
            #         # drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
            #         # drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
            #         # drugs = (drug_a, drug_b)
            #
            #         if epoch == setting.n_epochs - 1:
            #
            #             #### save intermediate results in the last traing epoch
            #             preds = best_drug_model(*local_batch)
            #             cur_train_start_index = setting.batch_size * (val_train_i - 1)
            #             cur_train_stop_index = min(setting.batch_size * (val_train_i), len(training_index_list))
            #             for n, m in best_drug_model.named_modules():
            #                 if n == "out":
            #                     catoutput = m._value_hook[0]
            #             for i, train_combination in enumerate(training_index_list[cur_train_start_index: cur_train_stop_index]):
            #
            #                 if not path.exists("train_" + setting.catoutput_output_type + "_datas"):
            #                     mkdir("train_" + setting.catoutput_output_type + "_datas")
            #                 save(catoutput.narrow_copy(0,i,1), path.join("train_" + setting.catoutput_output_type + "_datas",
            #                                                            str(train_combination) + '.pt'))
            #                 save_data_num += 1
            #
            #         # preds = drug_model(*local_batch, drugs = drugs)
            #         # Model computations
            #         preds = drug_model(*local_batch)
            #         preds = preds.contiguous().view(-1)
            #         preds = torch.sigmoid(preds)
            #         assert preds.size(0) == local_labels.size(0)
            #         prediction_on_cpu = preds.cpu().numpy()
            #         # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
            #         #                                   prediction_on_cpu[sample_size:]], axis=0)
            #         mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
            #         all_preds.append(mean_prediction_on_cpu)
            #         all_ys.append(local_labels_on_cpu)

            #    logger.debug("saved {!r} data points".format(save_data_num))
            #     all_preds = np.concatenate(all_preds)
            #     all_ys = np.concatenate(all_ys).astype(int)
            #     assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
            #     val_train_roc_auc = roc_auc_score(all_ys.reshape(-1), all_preds.reshape(-1))
            #     val_train_pr_auc = average_precision_score(all_ys.reshape(-1), all_preds.reshape(-1))
            #     if epoch == setting.n_epochs - 1 and setting.save_final_pred:
            #         save(np.concatenate((np.array(training_index_list).reshape(-1,1), all_preds.reshape(-1,1), all_ys.reshape(-1,1)), axis=1), "prediction/prediction_" + setting.catoutput_output_type + "_training")
            #
                val_i = 0
                all_preds = []
                all_ys = []

                for (local_batch, smiles_a, smiles_b), local_labels in validation_generator:

                    val_i += 1
                    local_labels_on_cpu = np.array(local_labels).reshape(-1)
                    sample_size = local_labels_on_cpu.shape[-1]
                    local_labels_on_cpu = local_labels_on_cpu[:sample_size].astype(int)
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                    reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices)+ setting.single_repsonse_feature_length))
                    local_batch = reorder_tensor.get_reordered_narrow_tensor()
                    # drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
                    # drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
                    # drugs = (drug_a, drug_b)
                    # preds = drug_model(*local_batch, drugs = drugs)
                    # Model computations
                    preds = drug_model(*local_batch)
                    preds = preds.contiguous().view(-1)
                    preds = torch.sigmoid(preds)
                    assert preds.size(0) == local_labels.size(0)
                    prediction_on_cpu = preds.cpu().numpy()
                    # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                    #                                   prediction_on_cpu[sample_size:]], axis=0)
                    mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                    all_preds.append(mean_prediction_on_cpu)
                    all_ys.append(local_labels_on_cpu)

                all_preds = np.concatenate(all_preds)
                all_ys = np.concatenate(all_ys)

                assert len(all_preds) == len(all_ys), "predictions and labels are in different length"

                val_roc_auc = roc_auc_score(all_ys.reshape(-1), all_preds.reshape(-1))
                val_pr_auc = average_precision_score(all_ys.reshape(-1), all_preds.reshape(-1))

                if best_auc < val_pr_auc:
                    best_auc = val_pr_auc
                    best_drug_model.load_state_dict(drug_model.state_dict())

            scheduler.step()

            logger.debug("Training roc_auc is {0!r}, Training pr_auc is {1!r}".format(val_train_roc_auc, val_train_pr_auc))
            logger.debug("Validation roc_auc is {0!r}, Validation pr_auc is {1!r}".format(val_roc_auc, val_pr_auc))
            if USE_wandb:
                wandb.log({"Training roc_auc": val_train_roc_auc, "Training pr_auc": val_train_pr_auc}, step=epoch)
                wandb.log({"Validation roc_auc": val_roc_auc, "Validation pr_auc": val_pr_auc}, step=epoch)

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    save_data_num = 0

    with torch.set_grad_enabled(False):

        best_drug_model.eval()
        all_preds = []
        all_ys = []
        for (local_batch, smiles_a, smiles_b), local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1]
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
            reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length))
            local_batch = reorder_tensor.get_reordered_narrow_tensor()
            # drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
            # drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
            # drugs = (drug_a, drug_b)
            # preds = drug_model(*local_batch, drugs = drugs)
            # Model computations
            preds = drug_model(*local_batch)
            preds = preds.contiguous().view(-1)
            preds = torch.sigmoid(preds)
            cur_test_start_index = test_params['batch_size'] * (test_i-1)
            cur_test_stop_index = min(test_params['batch_size'] * (test_i), len(test_index_list))
            for n, m in best_drug_model.named_modules():
                if n == "out":
                    catoutput = m._value_hook[0]
            for i, test_combination in enumerate(test_index_list[cur_test_start_index: cur_test_stop_index]):
                if not path.exists("test_" + setting.catoutput_output_type + "_datas"):
                    mkdir("test_" + setting.catoutput_output_type + "_datas")
                save(catoutput.narrow_copy(0, i, 1), path.join("test_" + setting.catoutput_output_type + "_datas",
                                                             str(test_combination) + '.pt'))
                save_data_num += 1
            assert preds.size(0) == local_labels.size(0)
            prediction_on_cpu = preds.cpu().numpy()
            mean_prediction_on_cpu = prediction_on_cpu
            all_preds.append(mean_prediction_on_cpu)
            all_ys.append(local_labels_on_cpu)

        logger.debug("saved {!r} data for testing dataset".format(save_data_num))
        all_preds = np.concatenate(all_preds)
        all_ys = np.concatenate(all_ys).astype(int)
        assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
        test_roc_auc = roc_auc_score(all_ys.reshape(-1), all_preds.reshape(-1))
        test_pr_auc = average_precision_score(all_ys.reshape(-1), all_preds.reshape(-1))
        save(np.concatenate((np.array(test_index_list).reshape(-1,1), all_preds.reshape(-1, 1), all_ys.reshape(-1, 1)), axis=1),
             "prediction/prediction_" + setting.catoutput_output_type + "_testing")

        avg_loss = test_total_loss
        test_loss.append(avg_loss)

    logger.debug("Testing mse is {0}, Testing roc_auc is {1!r}, Testing pr_auc is {2!r}".format(np.mean(test_loss), test_roc_auc, test_pr_auc))

    batch_input_importance = []
    batch_out_input_importance = []
    batch_transform_input_importance = []
    total_data, _ = next(iter(eval_train_shap_generator))
    total_data = total_data.float().to(device2)
    total_data = total_data.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
    reorder_tensor.load_raw_tensor(total_data)
    total_data = reorder_tensor.get_reordered_narrow_tensor()
    for (local_batch, smiles_a, smiles_b), local_labels in test_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
        local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
        reorder_tensor.load_raw_tensor(local_batch)
        local_batch = reorder_tensor.get_reordered_narrow_tensor()
        drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
        drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
        drugs = (drug_a, drug_b)
        if setting.save_feature_imp_model:
            save(best_drug_model, setting.best_model_path)
        # Model computations
        logger.debug("Start feature importances analysis")
        if setting.save_easy_input_only:
            e = shap.GradientExplainer(best_drug_model, data=list(total_data))
            input_importance = e.shap_values(list(local_batch))
            #pickle.dump(input_shap_values, open(setting.input_importance_path, 'wb+'))
        else:
            input_importance = []
            for layer in best_drug_model.linear_layers:
                cur_e = shap.GradientExplainer((best_drug_model, layer), data=list(total_data))
                cur_input_importance = cur_e.shap_values(list(local_batch))
                input_importance.append(cur_input_importance)
            input_importance = np.concatenate(tuple(input_importance), axis=1)
        batch_input_importance.append(input_importance)
        logger.debug("Finished one batch of input importance analysis")

        e1 = shap.GradientExplainer((best_drug_model, best_drug_model.out), data=list(total_data))
        out_input_shap_value = e1.shap_values(list(local_batch))
        batch_out_input_importance.append(out_input_shap_value)
        logger.debug("Finished one batch of out input importance analysis")

        if setting.save_inter_imp:
            transform_input_importance = []
            for layer in best_drug_model.dropouts:
                cur_e = shap.GradientExplainer((best_drug_model, layer), data=list(total_data))
                cur_transform_input_shap_value = cur_e.shap_values(list(local_batch))
                transform_input_importance.append(cur_transform_input_shap_value)
            transform_input_importance = np.concatenate(tuple(transform_input_importance), axis=1)

            batch_transform_input_importance.append(transform_input_importance)
        logger.debug("Finished one batch of importance analysis")
    batch_input_importance = np.concatenate(tuple(batch_input_importance), axis=0)
    batch_out_input_importance = np.concatenate(tuple(batch_out_input_importance), axis=0)
    pickle.dump(batch_input_importance, open(setting.input_importance_path, 'wb+'))
    pickle.dump(batch_out_input_importance, open(setting.out_input_importance_path, 'wb+'))
    if setting.save_inter_imp:
        batch_transform_input_importance = np.concatenate(tuple(batch_transform_input_importance), axis=0)
        pickle.dump(batch_transform_input_importance, open(setting.transform_input_importance_path, 'wb+'))


if __name__ == "__main__":

    try:
        run()
        logger.debug("new directory %s" % setting.run_dir)

    except:

        import shutil

        shutil.rmtree(setting.run_dir)
        logger.debug("clean directory %s" % setting.run_dir)
        raise