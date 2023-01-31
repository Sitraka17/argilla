#import argila as rg 

#rg.init(api_url=url)


import pandas as pd
import argilla as rg
from datasets import load_dataset

# load dataset from the hub
#dataset = load_dataset("argilla/gutenberg_spacy-ner", split="train")

# read in dataset, assuming its a dataset for token classification
#dataset_rg = rg.read_datasets(dataset, task="TokenClassification")

# log the dataset
#rg.log(dataset_rg, "gutenberg_spacy-ner")

# load dataset from json
my_dataframe = pd.read_json(
    "https://raw.githubusercontent.com/recognai/datasets/main/sst-sentimentclassification.json")

# convert pandas dataframe to DatasetForTextClassification
dataset_rg = rg.DatasetForTextClassification.from_pandas(my_dataframe)

# log the dataset
rg.log(dataset_rg, name="sst-sentimentclassification")
