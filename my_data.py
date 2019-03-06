import setting
import pandas as pd
import random_test
import os

import torch
from torch.utils import data

class CustomDataLoader:
    pass

class ExpressionDataLoader(CustomDataLoader):
    gene_expression = None
    backup_expression = None

    def __init__(self):
        super()

    @classmethod
    def initialize_gene_expression(cls):
        ### make sure only one gene expression data frame is instantiated in this class
        ### return: gene expression data frame

        if cls.gene_expression is None:
            cls.gene_expression = pd.read_csv(setting.gene_expression, sep='\t')
            random_test.logger.debug("Read in gene expresion data successfully")
            cls.gene_expression.set_index(keys='Entrez', inplace=True)
        return cls.gene_expression

    @classmethod
    def initialize_backup_expression(cls):
        ### make sure only one backup expression data frame is instantiated in this class
        ### data will firstly searched in CCLE database and then in GDSC database
        ### return: gene expression data frame

        if cls.backup_expression is None:
            cls.backup_expression = pd.read_csv(setting.backup_expression, sep='\t')
            random_test.logger.debug("Read in back up expresion data successfully")
            cls.backup_expression.set_index(keys='Entrez', inplace=True)
        return cls.backup_expression

    def __filter_genes(self, df, entrezIDs):

        ### genes: interested genes
        ### return data frame: Select only the genes interested in the data frame

        result_df = df.loc[entrezIDs, :]
        repo_genes, interested_genes = set(df.index), set(entrezIDs)
        if not repo_genes.issuperset(interested_genes):
            unfound = interested_genes - repo_genes
            random_test.logger.debug("{!r} are not found!".format(unfound))

        result_df.fillna(0, inplace=True)
        return result_df

    def __filter_celllines(self, df, celllines):

        ### cell line: interested cell lines
        ### return data frame: select only the cell lines interested by user
        result_df = df.loc[:, celllines]
        repo_celllines, interested_celllines, unfound = set(df.columns), set(celllines), {}

        if not repo_celllines.issuperset(interested_celllines):
            unfound = interested_celllines - repo_celllines
            random_test.logger.debug("{!r} are not found!".format(unfound))

        if len(unfound):
            ### use the back up expression dataframe data
            self.initialize_backup_expression()
            backup_celllines_repo = set(self.backup_expression.columns)
            if len(unfound.intersection(backup_celllines_repo)):
                more_cellline_df = self.__filter_celllines(self.backup_expression, list(unfound))
                result_df = pd.concat([result_df.drop(columns = list(unfound)), more_cellline_df], axis=1)

        result_df.fillna(0, inplace=True)
        return result_df

    def prepare_expresstion_df(self, entrezIDs, celllines):

        ### entrezIDs, celllines: selection criterials
        ### return data frame: data frame that have interested cell lines and genes
        ###              A375   ..... (celllines)
        ###   1003(entrez)
        ###    ...

        self.initialize_gene_expression()

        result_df = self.__filter_celllines(self.gene_expression, celllines)
        result_df = self.__filter_genes(result_df, entrezIDs)
        if setting.expression_data_renew or not os.path.exists(setting.processed_expression):
            random_test.logger.debug("Persist gene expression data frame")
            result_df.to_csv(setting.processed_expression, index = False)

        return result_df

class MyDataset(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        drug_combine_file = 'datas/' + ID + '.pt'
        # Load data and get label
        try:
            X = torch.load(drug_combine_file)
        except:
            random_test.logger.error("Fail to get {}".format(ID))
            raise
        y = self.labels[ID]

        return X, y