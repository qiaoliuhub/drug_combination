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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch_visual

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

    synergy_score = my_data.SynergyDataReader.get_synergy_score()

    final_index_1 = synergy_score.reset_index().apply(lambda row: row['drug_a_name']+'_'+row['drug_b_name']+'_' +
                                               row['cell_line'] + '_' + str(row['index']), axis = 1)
    final_index_2 = synergy_score.reset_index().apply(lambda row: row['drug_b_name']+'_'+row['drug_a_name']+'_' +
                                               row['cell_line'] + '_' + str(row['index']), axis = 1)
    final_index = pd.concat([final_index_1, final_index_2], axis = 0).reset_index(drop=True)

    # print(simulated_drug_target, simulated_drug_target.shape)
    # print("synergy_score filtered data amount %s" %str(len(synergy_score)))
    # print(simulated_drug_target.shape, sel_dp.shape)
    # print(sel_dp)
    # print(sel_dp.shape)
    std_scaler = StandardScaler()
    logger.debug("Getting features and synergy scores ...")
    X, drug_features_len, cl_features_len, drug_features_name, cl_features_name = \
        my_data.SamplesDataLoader.Raw_X_features_prep(methods='attn')
    Y = my_data.SamplesDataLoader.Y_features_prep()
    logger.debug("Spliting data ...")

    # train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
    #     my_data.DataPreprocessor.reg_train_eval_test_split()
    test_index, test_index_2, test_generator = None, None, None
    cv_pearson_scores = []
    cv_models = []

    split_fun = my_data.DataPreprocessor.reg_train_eval_test_split

    for train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 in split_fun():

        local_X = X[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]
        final_index_for_X = final_index.iloc[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]

        std_scaler.fit(Y[train_index])
        if setting.y_transform:
            Y = std_scaler.transform(Y) * 100

        for i, combin_drug_feature_array in enumerate(local_X):
            if setting.unit_test:
                if i<=501:# and not os.path.exists(os.path.join('datas', str(final_index_for_X.iloc[i]) + '.pt')):
                    save(combin_drug_feature_array, os.path.join('datas', str(final_index_for_X.iloc[i]) + '.pt'))
            else:
                if not os.path.exists(os.path.join('datas', str(final_index_for_X.iloc[i]) + '.pt')):
                    save(combin_drug_feature_array, os.path.join('datas', str(final_index_for_X.iloc[i]) + '.pt'))

        partition = {'train': list(final_index.iloc[train_index]),
                     'test1': list(final_index.iloc[test_index]), 'test2': list(final_index.iloc[test_index_2]),
                     'eval1': list(final_index.iloc[evaluation_index]),
                     'eval2': list(final_index.iloc[evaluation_index_2])}

        labels = {key: value for key, value in zip(list(final_index),
                                                   list(Y.reshape(-1)))}

        train_params = {'batch_size': setting.batch_size,
                        'shuffle': True}
        eval_params = {'batch_size': len(test_index) * 2,
                       'shuffle': True}
        test_params = {'batch_size': len(test_index) * 2,
                       'shuffle': True}

        logger.debug("Preparing datasets ... ")
        #training_set = my_data.MyDataset(partition['train'], labels)
        training_set = my_data.MyDataset(partition['train'] + partition['eval1'] + partition['eval2'], labels)
        training_generator = data.DataLoader(training_set, **train_params)

        #validation_set = my_data.MyDataset(partition['eval1'] + partition['eval2'], labels)
        validation_set = my_data.MyDataset(partition['test1'] + partition['test2'], labels)
        validation_generator = data.DataLoader(validation_set, **eval_params)

        test_set = my_data.MyDataset(partition['test1'] + partition['test2'], labels)
        test_generator = data.DataLoader(test_set, **test_params)

        logger.debug("Preparing models")
        drug_model = attention_model.get_model()
        torchsummary.summary(drug_model, input_size=[(setting.n_feature_type, setting.d_input), (setting.n_feature_type, setting.d_input)])
        optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay, betas=(0.9, 0.98), eps=1e-9)

        logger.debug("Start training")
        # Loop over epochs
        mse_visualizer = torch_visual.VisTorch(env_name='MSE')
        pearson_visualizer = torch_visual.VisTorch(env_name='Pearson')

        for epoch in range(setting.n_epochs):

            drug_model.train()
            start = time()
            cur_epoch_train_loss = []
            train_total_loss = 0
            i = 0

            # Training
            for local_batch, local_labels in training_generator:
                i += 1
                # Transfer to GPU
                local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)

                # Model computations
                preds = drug_model(local_batch, local_batch).contiguous().view(-1)
                ys = local_labels.contiguous().view(-1)
                optimizer.zero_grad()
                assert preds.size(-1) == ys.size(-1)
                loss = F.mse_loss(preds, ys)
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()

                n_iter = 50
                if (i + 1) % n_iter == 0:
                    p = int(100 * (i + 1) / setting.batch_size)
                    avg_loss = train_total_loss / n_iter
                    if setting.y_transform:
                        avg_loss = std_scaler.inverse_transform(np.array(avg_loss/100).reshape(-1,1)).reshape(-1)[0]
                    random_test.logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                          ((time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, avg_loss))
                    train_total_loss = 0
                    cur_epoch_train_loss.append(avg_loss)

            ### Evaluation
            val_i = 0
            val_total_loss = 0
            val_loss = []
            val_pearson = 0

            with torch.set_grad_enabled(False):

                drug_model.eval()
                for local_batch, local_labels in validation_generator:
                    val_i += 1
                    local_labels_on_cpu = np.array(local_labels).reshape(-1)
                    sample_size = local_labels_on_cpu.shape[-1]//2
                    local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                    preds = drug_model(local_batch, local_batch).contiguous().view(-1)
                    assert preds.size(-1) == local_labels.size(-1)
                    prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                    mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                                                      prediction_on_cpu[sample_size:]], axis=0)
                    if setting.y_transform:
                        local_labels_on_cpu, mean_prediction_on_cpu = \
                            std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1,1) / 100), \
                            std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1,1) / 100)

                    loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
                    val_pearson = pearsonr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]
                    val_total_loss += loss

                    n_iter = 1
                    if (val_i + 1) % n_iter == 0:
                        avg_loss = val_total_loss / n_iter
                        val_loss.append(avg_loss)
                        val_total_loss = 0

                cv_pearson_scores.append(val_pearson)
                cv_models.append(drug_model)

            logger.debug(
                "Validation mse is {0}, Validation pearson correlation is {1!r}".format(np.mean(val_loss), val_pearson))
            mse_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss), np.mean(val_loss), loss_type='mse',
                                     ytickmin=100, ytickmax=500)
            pearson_visualizer.plot_loss(epoch, val_pearson, loss_type='pearson_loss', ytickmin=0, ytickmax=1)

    best_index = np.argmax(cv_pearson_scores)
    drug_model = cv_models[int(best_index)]

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_pearson = 0

    with torch.set_grad_enabled(False):

        drug_model.eval()
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1] // 2
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)

            # Model computations
            preds = drug_model(local_batch, local_batch).contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                                              prediction_on_cpu[sample_size:]], axis=0)
            if setting.y_transform:
                local_labels_on_cpu, mean_prediction_on_cpu = \
                    std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                    std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
            loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
            test_pearson = pearsonr(local_labels_on_cpu.reshape(-1), mean_prediction_on_cpu.reshape(-1))[0]
            test_total_loss += loss

            n_iter = 1
            if (test_i + 1) % n_iter == 0:
                avg_loss = test_total_loss / n_iter
                test_loss.append(avg_loss)
                test_total_loss = 0

    logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r}".format(np.mean(test_loss), test_pearson))
