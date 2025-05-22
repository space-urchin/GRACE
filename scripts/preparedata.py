import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split


def prepare_data(dataset_name, old_classes, new_classes):
    if dataset_name == "iot":
        data = pd.read_csv("../data/iot.csv")
        data = data[data["Label"] != "ARP Spoofing"]
        data.drop(["Flow ID", "Src IP", "Dst IP", "Timestamp"], axis=1, inplace=True)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        categorical_columns = ["Connection Type", "Src Port", "Dst Port", "Protocol"]
        numeric_columns = [col for col in data.columns if col not in categorical_columns + ["Label"]]

        X_num = data[numeric_columns].copy()
        y = data[["Label"]]

        X_cat_hashed = pd.DataFrame()
        for col in categorical_columns:
            col_data = data[col].astype(str)
            vectorizer = HashingVectorizer(n_features=32, alternate_sign=False)
            hashed_matrix = vectorizer.fit_transform(col_data)
            hashed_df = pd.DataFrame(
                hashed_matrix.toarray(),
                columns=[f"{col}_hash_{i}" for i in range(32)]
            )
            X_cat_hashed = pd.concat([X_cat_hashed, hashed_df], axis=1)

        X = pd.concat([X_num.reset_index(drop=True), X_cat_hashed.reset_index(drop=True)], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y["Label"], random_state=42
        )
        label_col = "Label"

    elif dataset_name == "cic":
        data = pd.read_csv('../data/cic.csv')
    
        X = data.drop(['Attack Type'], axis=1)
        y = data[['Attack Type']] 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,  # 80% train, 20% test
            stratify=y['Attack Type'],  # Maintain class distribution
            random_state=42
        )

        label_col = "Attack Type"


    elif dataset_name == "unsw":
        train_df = pd.read_csv("../data/UNSW_NB15_training-set.csv")
        test_df = pd.read_csv("../data/UNSW_NB15_testing-set.csv")

        for cat in ['Worms', 'Backdoor', 'Analysis']:
            train_df = train_df[train_df['attack_cat'] != cat]
            test_df = test_df[test_df['attack_cat'] != cat]

        train_df['service'].replace('-', np.nan, inplace=True)
        test_df['service'].replace('-', np.nan, inplace=True)
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        drop_columns = ["id", "label"]
        train_df = train_df.drop(drop_columns, axis=1)
        test_df = test_df.drop(drop_columns, axis=1)

        categorical_cols = ["proto", "state", "service"]
        train_df, test_df = preprocess_categorical_columns(train_df, test_df, categorical_cols)

        X_train = train_df.drop(["attack_cat"], axis=1)
        y_train = train_df[["attack_cat"]]
        X_test = test_df.drop(["attack_cat"], axis=1)
        y_test = test_df[["attack_cat"]]
        label_col = "attack_cat"

    else:
        raise ValueError("Unsupported dataset name")

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    id_mask = y_train[label_col].isin(old_classes)
    ood_mask = y_train[label_col].isin(new_classes)

    X_id = X_train[id_mask].copy()
    y_id = y_train[id_mask][label_col]
    X_ood = X_train[ood_mask].copy()
    y_ood = y_train[ood_mask][label_col]

    scaler = StandardScaler()
    X_id = scaler.fit_transform(X_id)
    X_ood = scaler.transform(X_ood)
    X_test = scaler.transform(X_test)

    return X_id, y_id, X_ood, y_ood, X_test, y_test, label_col


def preprocess_categorical_columns(train_df, test_df, categorical_cols):
    """
    Preprocess categorical columns using one-hot encoding
    """
    # Combine train and test for consistent encoding
    combined_df = pd.concat([train_df, test_df], axis=0)
    
    # One-hot encode all categorical columns
    encoded_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=False)
    
    # Split back into train and test
    train_encoded = encoded_df.iloc[:len(train_df)]
    test_encoded = encoded_df.iloc[len(train_df):]
    
    return train_encoded, test_encoded