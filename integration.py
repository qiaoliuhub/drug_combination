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
from scipy.stats import pearsonr, spearmanr
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

    if not setting.update_final_index and os.path.exists(setting.final_index):
        final_index = pd.read_csv(setting.final_index, header=None)[0]
    else:
        final_index = my_data.SynergyDataReader.get_final_index()

    logger.debug("Preparing models")
    drug_model = attention_model.get_retrain_model()
    best_drug_model = attention_model.get_retrain_model()
    # torchsummary.summary(drug_model, input_size=[(setting.n_feature_type, setting.d_input), (setting.n_feature_type, setting.d_input)])
    optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min = 1e-7)
    # train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
    #     my_data.DataPreprocessor.reg_train_eval_test_split()

    logger.debug("Preparing Data")
    if len(setting.catoutput_intput_type) and os.path.exists("train_" + setting.catoutput_intput_type[0] + "_datas"):
        training_index = [f[:-3] for f in os.listdir("train_" + setting.catoutput_intput_type[0] + "_datas")]
        test_index = [f[:-3] for f in os.listdir("test_" + setting.catoutput_intput_type[0] + "_datas")]
        save(training_index, setting.train_index)
        save(test_index, setting.test_index)
    else:
        training_index = load(setting.train_index)
        test_index = load(setting.test_index)
    training_index = [x for x in training_index if x in set(final_index)]
    test_index = [x for x in test_index if x in set(final_index)]
    logger.debug("Training data length: {!r}".format(len(training_index)))
    logger.debug("Testing data length: {!r}".format(len(test_index)))
    y_labels = load(setting.y_labels_file)
    std_scaler = StandardScaler()
    if setting.y_transform:
        std_scaler.fit(np.array(list(y_labels.values())).reshape(-1, 1))
        for key in y_labels.keys():
            y_labels[key] = std_scaler.transform([[y_labels[key]]])[0,0]

    partition = {'train': training_index, 'test': test_index}

    train_params = {'batch_size': setting.batch_size,
                    'shuffle': True}
    eval_train_params = {'batch_size': len(training_index),
                         'shuffle': False}
    test_params = {'batch_size': len(test_index),
                   'shuffle': False}

    logger.debug("Preparing datasets ... ")

    if setting.update_features:
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
            for fea_type in setting.dir_input_type:
                drug_a, drug_b, cell_line, _ = drug_combin.split("_")
                if fea_type == 'proteomics':
                    input_dir = os.path.join(str(fea_type) + "_datas", str(cell_line) + ".pt")
                    pro_array = torch.load(input_dir)
                    cur_tensor = torch.from_numpy(pro_array.reshape(1,-1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    continue

                if fea_type == 'single':
                    drug_a = "_".join([cell_line, drug_a])
                    drug_b = "_".join([cell_line, drug_b])
                input_dir_a = os.path.join(str(fea_type) + "_datas", str(drug_a) + ".pt")
                input_dir_b = os.path.join(str(fea_type) + "_datas", str(drug_b) + ".pt")
                drug_a_array = torch.load(input_dir_a)
                drug_b_array = torch.load(input_dir_b)
                if fea_type == 'single':
                    max_array = np.maximum(drug_a_array, drug_b_array)
                    min_array = np.minimum(drug_a_array, drug_b_array)
                    additive_drug = np.add(drug_a_array, drug_b_array)
                    drug_a_array = max_array
                    drug_b_array = min_array
                try:
                    cur_tensor = torch.from_numpy(drug_a_array.reshape(1,-1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    cur_tensor = torch.from_numpy(drug_b_array.reshape(1, -1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    if fea_type == 'single':
                        cur_tensor = torch.from_numpy(additive_drug.reshape(1, -1)).float().to(device2)
                        cur_tensor_list.append(cur_tensor)
                except:
                    random_test.logger.error("Fail to get {}".format(drug_combin))
                    raise
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
            for fea_type in setting.dir_input_type:
                drug_a, drug_b, cell_line, _ = drug_combin.split("_")
                if fea_type == 'proteomics':
                    input_dir = os.path.join(str(fea_type) + "_datas", str(cell_line) + ".pt")
                    pro_array = torch.load(input_dir)
                    cur_tensor = torch.from_numpy(pro_array.reshape(1,-1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    continue
                if fea_type == 'single':
                    drug_a = "_".join([cell_line, drug_a])
                    drug_b = "_".join([cell_line, drug_b])
                input_dir_a = os.path.join(str(fea_type) + "_datas", str(drug_a) + ".pt")
                input_dir_b = os.path.join(str(fea_type) + "_datas", str(drug_b) + ".pt")
                drug_a_array = torch.load(input_dir_a)
                drug_b_array = torch.load(input_dir_b)
                if fea_type == 'single':
                    max_array = np.maximum(drug_a_array, drug_b_array)
                    min_array = np.minimum(drug_a_array, drug_b_array)
                    additive_drug = np.add(drug_a_array, drug_b_array)
                    drug_a_array = max_array
                    drug_b_array = min_array
                try:
                    cur_tensor = torch.from_numpy(drug_a_array.reshape(1,-1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    cur_tensor = torch.from_numpy(drug_b_array.reshape(1, -1)).float().to(device2)
                    cur_tensor_list.append(cur_tensor)
                    if fea_type == 'single':
                        cur_tensor = torch.from_numpy(additive_drug.reshape(1, -1)).float().to(device2)
                        cur_tensor_list.append(cur_tensor)
                except:
                    random_test.logger.error("Fail to get {}".format(drug_combin))
                    raise
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
    best_cv_pearson_score = 0

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
        val_train_spearman = 0

        val_i = 0
        val_total_loss = 0
        val_loss = []
        val_pearson = 0
        val_spearman = 0

        with torch.set_grad_enabled(False):

            drug_model.eval()
            all_preds = []
            all_ys = []
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
                all_preds.append(mean_prediction_on_cpu)
                all_ys.append(local_labels_on_cpu)

            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)
            assert len(all_preds) == len(all_ys), "predictions and labels are in different length"

            val_train_loss = mean_squared_error(all_preds, all_ys)
            val_train_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            val_train_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            if epoch == setting.n_epochs - 1 and setting.save_final_pred:
                save(np.concatenate((np.array(training_index).reshape(-1,1), all_preds.reshape(-1,1), all_ys.reshape(-1,1)), axis=1),
                     "prediction/prediction_" + setting.catoutput_output_type + "_training")



            for local_batch, local_labels in validation_generator:

                all_preds = []
                all_ys = []
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
                all_preds.append(mean_prediction_on_cpu)
                all_ys.append(local_labels_on_cpu)

            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)
            assert len(all_preds) == len(all_ys), "predictions and labels are in different length"

            val_loss = mean_squared_error(all_preds, all_ys)
            val_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            val_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]

            if best_cv_pearson_score < val_pearson:
                best_cv_pearson_score = val_pearson
                best_drug_model.load_state_dict(drug_model.state_dict())

        logger.debug(
            "Training mse is {0}, Training pearson correlation is {1!r}, Training Spearman correlation is {2!r}".
                format(np.mean(val_train_loss), val_train_pearson, val_train_spearman))

        logger.debug(
            "Validation mse is {0}, Validation pearson correlation is {1!r}, Validation Spearman correlation is {2!r}".
                format(np.mean(val_loss), val_pearson, val_spearman))

        mse_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss),np.mean(val_loss), np.mean(val_train_loss), loss_type='mse',
                                 ytickmin=100, ytickmax=500)
        pearson_visualizer.plot_loss(epoch, val_train_pearson, val_pearson, loss_type='pearson_loss', ytickmin=0, ytickmax=1)

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_pearson = 0
    test_spearman = 0

    with torch.set_grad_enabled(False):

        best_drug_model.eval()
        all_preds = []
        all_ys = []
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1]
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
            # Model computations
            preds = best_drug_model(local_batch)
            preds = preds.contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                                              prediction_on_cpu[:sample_size]], axis=0)
            if setting.y_transform:
                local_labels_on_cpu, mean_prediction_on_cpu = \
                    std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                    std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
            all_preds.append(mean_prediction_on_cpu)
            all_ys.append(local_labels_on_cpu)


        all_preds = np.concatenate(all_preds)
        all_ys = np.concatenate(all_ys)
        assert len(all_preds) == len(all_ys), "predictions and labels are in different length"

        sample_size = len(all_preds)
        mean_prediction = np.mean([all_preds[:sample_size],
                                   all_preds[:sample_size]], axis=0)
        mean_y = np.mean([all_ys[:sample_size],
                          all_ys[:sample_size]], axis=0)

        test_loss = mean_squared_error(mean_y, mean_prediction)
        test_pearson = pearsonr(mean_y.reshape(-1), mean_prediction.reshape(-1))[0]
        test_spearman = spearmanr(mean_y.reshape(-1), mean_prediction.reshape(-1))[0]
        save(np.concatenate((np.array(test_index[:sample_size]).reshape(-1,1), mean_prediction.reshape(-1, 1), mean_y.reshape(-1, 1)), axis=1),
             "prediction/prediction_" + setting.catoutput_output_type + "_testing")


    logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r}, Testing spearman correlation is {2!r}".
                 format(np.mean(test_loss), test_pearson, test_spearman))