# Motivation

In many patients, a tumor’s innate or acquired resistance to a given therapy will render the treatment ineffective. To increase therapeutic options and to overcome drug resistance, cancer researchers have been actively investigating drug combinations.

## Build environment (dependencies)
```
conda env create -f environment.yml
```

## Check model performance with differnt cell line features (gene dependencies, gene expression and netexpress scores)
### gene dependencies
```
cp setting_gene_dependencies.py setting.py
```

### gene expression
```
cp setting_gene_expression.py setting.py
```

### netexpress
```
cp setting_net.py setting.py
```

# Run 
```
python ./attention_main.py
```
#### check results
check the logfile in the newest ```_run_*****``` folder

# Dataset

## Drug combination Synergy scores

#### AstraZeneca-Sanger Drug Combination Prediction DREAM Challenge

```
CELL_LINE,COMPOUND_A,COMPOUND_B,MAX_CONC_A,MAX_CONC_B,IC50_A,H_A,Einf_A,IC50_B,H_B,Einf_B,SYNERGY_SCORE,QA,COMBINATION_ID
BT549,ADAM17,AKT,1.0,75.0,0.184407667,1.7375109,59.57523324,1.0,0.0,100.0,,1.0,ADAM17.AKT
CAL148,ADAM17,AKT,1.0,75.0,0.153391224,1.7541229,1.10077945,1.0,0.0,100.0,,1.0,ADAM17.AKT
HCC38,ADAM17,AKT,1.0,75.0,1.0,10.0,64.40008038,1.0,0.0,100.0,,1.0,ADAM17.AKT
BT20,ADAM17,BCL2_BCL2L1,1.0,75.0,1.0,1.1617315,39.16458718,75.0,0.9657921,70.9150332,,1.0,ADAM17.BCL2_BCL2L1
HCC1143,ADAM17,BCL2_BCL2L1,1.0,75.0,0.115309253,10.0,57.19879477,0.0075,0.1,80.12808895,,1.0,ADAM17.BCL2_BCL2L1
HCC1937,ADAM17,BCL2_BCL2L1,1.0,75.0,1.0,0.6800246,44.77685179,1.460265179,0.8454861,93.47342327,,1.0,ADAM17.BCL2_BCL2L1
HS578T,ADAM17,BCL2_BCL2L1,1.0,75.0,0.11382074,1.8395583,62.84448784,1.188576126,0.2080817,78.96791015,,1.0,ADAM17.BCL2_BCL2L1
```

#### An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies. (O'Neil J et. al)

```
Unnamed: 0,drug_a_name,drug_b_name,cell_line,synergy
5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.6935301658
5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.7780530601
5-FU_ABT-888_A375,5-FU,ABT-888,A375,-1.1985054379
5-FU_ABT-888_A427,5-FU,ABT-888,A427,2.5956844375
5-FU_ABT-888_CAOV3,5-FU,ABT-888,CAOV3,-5.1399712212
5-FU_ABT-888_DLD1,5-FU,ABT-888,DLD1,1.9351271188
```

## Cell line gene-level dependencies score

```
index,genes,CAL148,HT1197,A2780,MCF7,HCC1395, ..., ENTREZID
CDH2 (1000),CDH2,0.1709,0.2125,0.1651,-0.0103,-0.2448, ..., 1000
AKT3 (10000),AKT3,-0.0400,-0.0137,0.2180,-0.0194],-0.0503644515648, ..., 10000
```

## Drug features


## Gene networks

```
Entrezid Entrezid posterior prob.
5988    53905   0.137373
5988    286234  0.116511
5988    277     0.104127
5988    387856  0.114427
5988    90317   0.115751
5988    100287366       0.100036
5988    100287362       0.105938
```
