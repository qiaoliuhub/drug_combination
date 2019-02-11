from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout
import setting
from keras.optimizers import Adam


class DrugsCombModel():

    def __init__(self, drug_a_features, drug_b_features, cl_genes_dp_features):

        self.drug_a_features = drug_a_features
        self.drug_b_features = drug_b_features
        self.cl_genes_dp_features = cl_genes_dp_features
        self.input_len = int(drug_a_features.shape[1]) + int(drug_b_features.shape[1] + int(cl_genes_dp_features.shape[1]))

    def __mlp_model(self, nodes_nums, input_len):

        model = Sequential()

        for i in xrange(len(nodes_nums)):

            if i == 0:
                model.add(Dense(nodes_nums[i], input_shape=(input_len,)))
            else:
                model.add(Dense(nodes_nums[i]))

            model.add(BatchNormalization())
            model.add(Activation(setting.activation_method[i%len(setting.activation_method)]))
            model.add(Dropout(rate=setting.dropout))

        model.add(Dense(1))

        return model

    @classmethod
    def compile_transfer_learning_model(cls, model):

        custimized_rmsprop = Adam(lr=setting.start_lr, decay=setting.lr_decay)
        model.compile(optimizer=custimized_rmsprop, loss='mse', metrics=['mse'])
        return model

    def get_model(self, method = setting.model_type):

        crispr_model = getattr(self, "_{!s}__{!s}_model".format(self.__class__.__name__, method), self.__mlp_model)(setting.FC_layout, self.input_len)
        return self.compile_transfer_learning_model(crispr_model)