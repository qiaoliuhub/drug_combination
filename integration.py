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
import torch.nn as nn
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

    final_index = my_data.SynergyDataReader.get_final_index()

    logger.debug("Preparing models")
    drug_model = attention_model.get_retrain_model()
    # torchsummary.summary(drug_model, input_size=[(setting.n_feature_type, setting.d_input), (setting.n_feature_type, setting.d_input)])
    optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min = 1e-7)
    # train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
    #     my_data.DataPreprocessor.reg_train_eval_test_split()

    logger.debug("Preparing Data")
    training_index = [f[:-3] for f in os.listdir("train_" + setting.catoutput_intput_type[0] + "_datas")]
    test_index = [f[:-3] for f in os.listdir("test_" + setting.catoutput_intput_type[0] + "_datas")]
    y_labels = load(setting.y_labels_file)
    std_scaler = StandardScaler()
    if setting.y_transform:
        std_scaler.fit(np.array(list(y_labels.values())).reshape(-1, 1))
        for key in y_labels.keys():
            y_labels[key] = std_scaler.transform([[y_labels[key]]])[0,0]
    cv_pearson_scores = []
    cv_models = []

    partition = {'train': training_index, 'test': test_index}

    train_params = {'batch_size': setting.batch_size,
                    'shuffle': True}
    eval_train_params = {'batch_size': len(training_index),
                         'shuffle': False}
    test_params = {'batch_size': len(test_index),
                   'shuffle': False}

    logger.debug("Preparing datasets ... ")
    for drug_combin in training_index:
        cur_tensor_list = []
        for fea_type in setting.catoutput_intput_type:
            input_dir = os.path.join("train_" + str(fea_type) + "_datas", str(drug_combin) + ".pt")
            try:
                cur_tensor = torch.load(input_dir)
                cur_tensor_list.append(cur_tensor)
            except:
                random_test.logger.error("Fail to get {}".format(drug_combin))
                raise
        save_path = "train_datas"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        final_tensor = torch.cat(tuple(cur_tensor_list), dim=1)
        save(final_tensor.contiguous().view((final_tensor.size(1),-1)), os.path.join(save_path, str(drug_combin) + ".pt"))

    for drug_combin in test_index:
        cur_tensor_list = []
        for fea_type in setting.catoutput_intput_type:
            input_dir = os.path.join("test_" + str(fea_type) + "_datas", str(drug_combin) + ".pt")
            try:
                cur_tensor = torch.load(input_dir)
                cur_tensor_list.append(cur_tensor)
            except:
                random_test.logger.error("Fail to get {}".format(drug_combin))
                raise
        save_path = "test_datas"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        final_tensor = torch.cat(tuple(cur_tensor_list), dim=1)
        save(final_tensor.contiguous().view((final_tensor.size(1),-1)), os.path.join(save_path, str(drug_combin) + ".pt"))

    training_set = my_data.MyDataset(partition['train'], y_labels, prefix="train")
    training_generator = data.DataLoader(training_set, **train_params)

    eval_train_set = my_data.MyDataset(partition['train'], y_labels, prefix='train')
    eval_train_generator = data.DataLoader(eval_train_set, **eval_train_params)

    #validation_set = my_data.MyDataset(partition['eval1'] + partition['eval2'], labels)
    validation_set = my_data.MyDataset(partition['test'], y_labels, prefix='test')
    validation_generator = data.DataLoader(validation_set, **test_params)

    test_set = my_data.MyDataset(partition['test'], y_labels, prefix='test')
    test_generator = data.DataLoader(test_set, **test_params)

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
            preds = drug_model(local_batch)
            preds = preds.contiguous().view(-1)
            ys = local_labels.contiguous().view(-1)
            optimizer.zero_grad()
            assert preds.size(-1) == ys.size(-1)
            loss = F.mse_loss(preds, ys)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

            n_iter = 2
            if i % n_iter == 0:
                sample_size = len(training_index)
                p = int(100 * i * setting.batch_size/sample_size)
                avg_loss = train_total_loss / n_iter
                if setting.y_transform:
                    avg_loss = std_scaler.inverse_transform(np.array(avg_loss / 100).reshape(-1, 1)).reshape(-1)[0]
                random_test.logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time() - start) // 60, epoch, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss))
                train_total_loss = 0
                cur_epoch_train_loss.append(avg_loss)

        #scheduler.step()

        ### Evaluation
        val_train_i = 0
        val_train_total_loss = 0
        val_train_loss = []
        val_train_pearson = 0

        val_i = 0
        val_total_loss = 0
        val_loss = []
        val_pearson = 0

        with torch.set_grad_enabled(False):

            drug_model.eval()
            for local_batch, local_labels in eval_train_generator:
                val_train_i += 1
                local_labels_on_cpu = np.array(local_labels).reshape(-1)
                sample_size = local_labels_on_cpu.shape[-1]
                local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                # Transfer to GPU
                local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                preds = drug_model(local_batch)
                preds = preds.contiguous().view(-1)
                assert preds.size(-1) == local_labels.size(-1)
                prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                #                                   prediction_on_cpu[sample_size:]], axis=0)
                mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                if setting.y_transform:
                    local_labels_on_cpu, mean_prediction_on_cpu = \
                        std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                        std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)

                loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
                val_train_pearson = pearsonr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]
                val_train_total_loss += loss

                n_iter = 1
                if val_train_i % n_iter == 0:
                    avg_loss = val_train_total_loss / n_iter
                    val_train_loss.append(avg_loss)
                    val_train_total_loss = 0

            for local_batch, local_labels in validation_generator:
                val_i += 1
                local_labels_on_cpu = np.array(local_labels).reshape(-1)
                sample_size = local_labels_on_cpu.shape[-1]
                local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                # Transfer to GPU
                local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                preds = drug_model(local_batch)
                preds = preds.contiguous().view(-1)
                assert preds.size(-1) == local_labels.size(-1)
                prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                #                                   prediction_on_cpu[sample_size:]], axis=0)
                mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                if setting.y_transform:
                    local_labels_on_cpu, mean_prediction_on_cpu = \
                        std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1,1) / 100), \
                        std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1,1) / 100)

                loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
                val_pearson = pearsonr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]
                val_total_loss += loss

                n_iter = 1
                if val_i % n_iter == 0:
                    avg_loss = val_total_loss / n_iter
                    val_loss.append(avg_loss)
                    val_total_loss = 0

            cv_pearson_scores.append(val_pearson)
            cv_models.append(drug_model)

        logger.debug(
            "Training mse is {0}, Training pearson correlation is {1!r}".format(np.mean(val_train_loss), val_train_pearson))

        logger.debug(
            "Validation mse is {0}, Validation pearson correlation is {1!r}".format(np.mean(val_loss), val_pearson))

        mse_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss),np.mean(val_loss), np.mean(val_train_loss), loss_type='mse',
                                 ytickmin=100, ytickmax=500)
        pearson_visualizer.plot_loss(epoch, val_train_pearson, val_pearson, loss_type='pearson_loss', ytickmin=0, ytickmax=1)


    best_index = np.argmax(cv_pearson_scores)
    best_drug_model = cv_models[int(best_index)]

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_pearson = 0

    with torch.set_grad_enabled(False):

        best_drug_model.eval()
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1] // 2
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
            # Model computations
            preds = drug_model(local_batch)
            preds = preds.contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                                              prediction_on_cpu[:sample_size]], axis=0)
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