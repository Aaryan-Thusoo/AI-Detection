import sklearn
import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    df = pd.read_csv(DATA_DIR / "labelled_data.csv")
    train, temp = sklearn.model_selection.train_test_split(df, stratify=df["label"], test_size=0.3)
    test, val = sklearn.model_selection.train_test_split(temp, stratify=temp["label"], test_size=0.5)

    train.to_csv(DATA_DIR / "label_train.csv")
    test.to_csv(DATA_DIR / "label_test.csv")
    val.to_csv(DATA_DIR / "label_val.csv")

if __name__ == "__main__":
    main()