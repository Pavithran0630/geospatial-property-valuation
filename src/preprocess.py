import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):

    # Remove missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove price outliers
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df["price"] >= lower) &
            (df["price"] <= upper)]

    return df


if __name__ == "__main__":

    df = load_data("../data/kc_house_data.csv")

    df = clean_data(df)

    df.to_csv("../data/cleaned_house_data.csv", index=False)

    print("Dataset cleaned successfully!")