from dataset_loader import load_ade_datasets, load_cadec_dataset
from data_cleaning import preprocess_dataframe
from save_datasets import save_dataframe

if __name__ == "__main__":
    # Load datasets
    df_ade_class, df_ade_relation = load_ade_datasets()
    df_cadec_train, df_cadec_valid, df_cadec_test = load_cadec_dataset()
    
    # Preprocess datasets
    df_ade_class = preprocess_dataframe(df_ade_class, text_columns=['text'])
    df_ade_relation = preprocess_dataframe(df_ade_relation, text_columns=['text'])
    df_cadec_train = preprocess_dataframe(df_cadec_train, text_columns=['text'])
    df_cadec_valid = preprocess_dataframe(df_cadec_valid, text_columns=['text'])
    df_cadec_test = preprocess_dataframe(df_cadec_test, text_columns=['text'])
    
    # Save processed datasets
    save_dataframe(df_ade_class, "processed_ade_class.csv")
    save_dataframe(df_ade_relation, "processed_ade_relation.csv")
    save_dataframe(df_cadec_train, "processed_cadec_train.csv")
    save_dataframe(df_cadec_valid, "processed_cadec_valid.csv")
    save_dataframe(df_cadec_test, "processed_cadec_test.csv")
    
    print("All datasets have been processed and saved successfully!")
