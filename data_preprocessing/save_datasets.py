import pandas as pd

def save_dataframe(df, file_path):
    """
    Save DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Data saved successfully at: {file_path}")

if __name__ == "__main__":
    sample_data = {"text": ["Sample sentence 1", "Sample sentence 2"], "label": [1, 0]}
    df = pd.DataFrame(sample_data)
    save_dataframe(df, "processed_data.csv")