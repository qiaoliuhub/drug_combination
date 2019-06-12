from sklearn import metrics
import numpy as np
from scipy.stats import spearmanr
import torch
import setting

class Ranking(object):
    def __init__(self, names):
        self.names = names

    def _normalize(self, impt, fea_num):

        assert len(self.names) == fea_num, "names length is different from features numbers"
        impt = impt / sum(impt)
        impt = list(zip(impt, self.names, range(fea_num)))
        impt.sort(key=lambda x: -x[0])
        return impt


class InputPerturbationRank(Ranking):
    def __init__(self, names):
        super(InputPerturbationRank, self).__init__(names)

    def _raw_rank(self, rep, y, network, x):

        fea_num = 0
        for fea in x:
            fea_num += int(fea.shape[1])
        impt = np.zeros(fea_num)

        fea_index = 0
        for fea_dfs in x:
            for i in range(fea_dfs.shape[1]):
                hold = np.array(fea_dfs[:, i])
                for j in range(rep):
                    np.random.shuffle(fea_dfs[:, i])

                    # Handle both TensorFlow and SK-Learn models.
                    if 'tensorflow' in str(type(network)).lower():
                        pred = list(network.predict(x))
                    else:
                        pred = network.predict(x)

                    rmse = metrics.mean_squared_error(y, pred)
                    #spearman_correlation = spearmanr(y, pred)[0]
                    impt[fea_index] += (rmse - impt[fea_index]) / (j + 1)

                fea_index += 1
                fea_dfs[:, i] = hold

        return impt, fea_num

    def _rank_for_drug_model(self, rep, y, model, x, reorder_tensor, scaler):

        fea_num = 0
        for fea in x:
            fea_num += int(fea.shape[1])
        impt = np.zeros(fea_num)
        fea_index = 0
        for i in range(x.shape[2]):

            hold = x[:,:,i].clone()
            rand_index = [i for i in range(hold.size(0))]
            for j in range(rep):
                np.random.shuffle(rand_index)
                x[:,:,i] = x[:,:,1][rand_index]
                reorder_tensor.load_raw_tensor(x)
                new_x = reorder_tensor.get_reordered_narrow_tensor()
                preds, _ = model(new_x, new_x)
                preds = preds.contiguous().view(-1)
                assert preds.size(-1) == y.size(-1)
                prediction_on_cpu = preds.cpu().numpy().reshape(-1)

                if setting.y_transform:
                    local_labels_on_cpu, mean_prediction_on_cpu = \
                        scaler.inverse_transform(y.reshape(-1, 1) / 100), \
                        scaler.inverse_transform(prediction_on_cpu.reshape(-1, 1) / 100)
                rmse = metrics.mean_squared_error(y, prediction_on_cpu)
                # spearman_correlation = spearmanr(y, pred)[0]
                impt[fea_index] += (rmse - impt[fea_index]) / (j + 1)

            fea_index += 1
            x[:, :, i] = hold

        return impt, fea_num

    def rank(self, rep, y, model, x, drug_model = False, reorder_tensor = None, scaler = None):
        if drug_model:
            assert reorder_tensor is not None and scaler is not None, "reorder_tensor and scaler should be used"
            impt, fea_num = self._rank_for_drug_model(rep, y, model, x, reorder_tensor, scaler)
        else:
            impt, fea_num = self._raw_rank(rep, y, model, x)
        return self._normalize(impt, fea_num)

