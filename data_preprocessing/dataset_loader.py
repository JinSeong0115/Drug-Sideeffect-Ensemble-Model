from datasets import load_dataset
import pandas as pd

def load_ade_datasets():
    """
    Load ADE Corpus V2 datasets and return as pandas DataFrames.
    """
    dc = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification")
    dr = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
    
    df_classification = pd.DataFrame(dc['train'])
    df_relation = pd.DataFrame(dr['train'])
    
    return df_classification, df_relation

def load_cadec_dataset():
    """
    Load CADEC dataset and return as pandas DataFrames.
    """
    ds = load_dataset("mireiaplalis/processed_cadec")
    df_train = pd.DataFrame(ds['train'])
    df_valid = pd.DataFrame(ds['validation'])
    df_test = pd.DataFrame(ds['test'])
    
    return df_train, df_valid, df_test

if __name__ == "__main__":
    df_ade_class, df_ade_relation = load_ade_datasets()
    df_cadec_train, df_cadec_valid, df_cadec_test = load_cadec_dataset()
    print("Datasets Loaded Successfully!")