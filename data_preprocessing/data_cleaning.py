import pandas as pd
import ast
import re

def safe_literal_eval(val):
    """
    Safely evaluate a string containing a Python literal structure (e.g., list, dict).
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def clean_text(text):
    """
    Remove special characters and extra spaces from text.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df, text_columns=[]):
    """
    Apply cleaning functions to DataFrame columns.
    """
    for col in text_columns:
        df[col] = df[col].astype(str).apply(clean_text)
    return df

if __name__ == "__main__":
    sample_data = {'text': ["Hello!! This is an example...", "Another!! Example??"]}
    df = pd.DataFrame(sample_data)
    df_cleaned = preprocess_dataframe(df, text_columns=['text'])
    print("Cleaned DataFrame:")
    print(df_cleaned)