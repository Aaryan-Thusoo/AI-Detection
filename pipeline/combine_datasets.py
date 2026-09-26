import pandas as pd
import pathlib

from helpers.searching import check_string

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

def combine_datasets() -> pd.DataFrame:
    phish_df = pd.read_csv(DATA_DIR / "raw" / "phishing_email.csv")
    phish_df["scam_type"] = "phishing"
    phish_df["ai_generated"] = 0

    ai_gen_df = pd.read_csv(DATA_DIR / "raw" / "ai_generated_scams.csv")

    df = pd.concat([phish_df, ai_gen_df], ignore_index=True)
    return df

def main():
    df = combine_datasets()

    df["Features"] = df["text_combined"].apply(check_string)
    df = df.dropna()

    df.to_csv(DATA_DIR / "processed" / "labelled_data.csv", index=False)


if __name__ == "__main__":
    main()
