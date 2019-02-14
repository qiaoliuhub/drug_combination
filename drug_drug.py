import pandas as pd
import setting
from sklearn.model_selection import ShuffleSplit
from scipy.stats import pearsonr
import logging
import os
import pickle

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def regular_split(df, group_col=None, n_split = 10, rd_state = setting.split_random_seed):

    shuffle_split = ShuffleSplit(test_size=1.0/n_split, random_state = rd_state)
    return shuffle_split.split(df).next()

def split_data(df):

    logger.debug("Splitting dataset to training dataset and testing dataset based on genes")
    if not setting.index_renewal and (os.path.exists(setting.train_index) and os.path.exists(setting.test_index)):
        train_index = pickle.load(open(setting.train_index, "rb"))
        test_index = pickle.load(open(setting.test_index, "rb"))
    else:
        train_index, test_index = regular_split(df)

        with open(setting.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
        with open(setting.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)

    logger.debug("Splitted data successfully")

    return train_index, test_index

def __ml_train_model():

    if setting.estimator == 'GradientBoosting':

        from h2o.estimators import H2OGradientBoostingEstimator

        rf_drugs = H2OGradientBoostingEstimator(
            model_id="rf_drugs",
            stopping_rounds=3,
            score_each_iteration=True,
            seed=10
        )

    # setting.estimator == 'RandomForest'
    else:

        from h2o.estimators import H2ORandomForestEstimator

        rf_drugs = H2ORandomForestEstimator(
            model_id="rf_drugs",
            stopping_rounds=3,
            score_each_iteration=True,
            seed=10)

    return rf_drugs

def __ml_train(X, y, train_index, test_index):

    import h2o

    try:
        logger.debug("Creating h2o working environment")
        # ### Start H2O
        # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:
        h2o.init(max_mem_size="10G", nthreads=10)
        h2o.remove_all()
        logger.debug("Created h2o working environment successfully")

        pre_h2o_df = pd.concat([X, y], axis=1)
        train_sample_size = 300 if setting.test_ml_train else len(train_index)
        h2o_drugs_train = h2o.H2OFrame(pre_h2o_df.loc[train_index[:train_sample_size], :])
        h2o_drugs_test = h2o.H2OFrame(pre_h2o_df.loc[test_index, :])

        logger.debug("Training machine learning model")
        rf_drugs = __ml_train_model()
        rf_drugs.train(x=h2o_drugs_train.col_names[:-1], y=h2o_drugs_train.col_names[-1],
                        training_frame=h2o_drugs_train)
        logger.debug("Trained successfully")

        logger.debug("Predicting training data")
        test_prediction_train = rf_drugs.predict(h2o_drugs_train[:-1])
        performance = pearsonr(test_prediction_train.as_data_frame()['predict'], h2o_drugs_train.as_data_frame()['synergy'])[0]
        logger.debug("spearman correlation coefficient for training dataset is: %f" % performance)

        logger.debug("Predicting test data")
        test_prediction = rf_drugs.predict(h2o_drugs_test[:-1])
        performance = pearsonr(test_prediction.as_data_frame()['predict'], h2o_drugs_test.as_data_frame()['synergy'])[0]
        logger.debug("spearman correlation coefficient for test dataset is: %f" % performance)

    except:
        raise

    finally:
        h2o.h2o.cluster().shutdown()
