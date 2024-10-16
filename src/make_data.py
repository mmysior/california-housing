#pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint: disable=invalid-name

#---------------- Imports ----------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler

def preprocess_data(raw_data_path: str, output_folder: str) -> None:
    #---------------- Data Loading ----------------
    housing = pd.read_csv(raw_data_path)

    len_orig = housing.shape[0]
    print(f'Number of examples in the dataset: {len_orig}')
    print(f'Number of features in the dataset: {housing.shape[1]}')
    print('---------------------------------------')

    #---------------- Data Preprocessing ----------------
    # Process missing data
    threshold = len(housing) * 0.05
    cols_to_drop = housing.columns[housing.isna().sum() <= threshold]
    housing.dropna(subset=cols_to_drop, inplace=True)

    house_value_cap = housing['median_house_value'].max()
    house_age_cap = housing['housing_median_age'].max()

    housing = housing[
        (housing['median_house_value'] < house_value_cap) &
        (housing['housing_median_age'] < house_age_cap)
    ]

    len_eda = housing.shape[0]

    print(f'Number of examples in the processed dataset: {len_eda}')
    print(f'Percentage of the original dataset: {(100 * len_eda / len_orig):.2f}%')
    print('---------------------------------------')

    #---------------- Feature Engineering ----------------
    # Create an income category for stratification
    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    #---------------- Train-Test Split ----------------
    strat_train_set, strat_test_set = train_test_split(
        housing,
        test_size=0.2,
        stratify=housing['income_cat'],
        random_state=1
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    housing = strat_train_set.copy()

    X_train = strat_train_set.drop("median_house_value", axis=1)
    Y_train = strat_train_set["median_house_value"].copy()

    X_test = strat_test_set.drop("median_house_value", axis=1)
    Y_test = strat_test_set["median_house_value"].copy()

    #---------------- Feature Preprocessing ----------------
    log_pipeline = make_pipeline(
        KNNImputer(),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        MinMaxScaler(feature_range=(0, 1)))

    one_hot_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder())

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        MinMaxScaler(feature_range=(0, 1)))

    preprocessing = ColumnTransformer(
        [
            ("log", log_pipeline, ['total_rooms', 'total_bedrooms', 'population', 'households']),
            ("one_hot", one_hot_pipeline, ["ocean_proximity"]),
        ],
    remainder=default_num_pipeline,
    )

    # Fit the preprocessing pipeline on the training data
    preprocessing.fit(X_train)

    # Transform both training and test data
    X_train_preprocessed = preprocessing.transform(X_train)
    X_test_preprocessed = preprocessing.transform(X_test)

    #---------------- Save Preprocessed Data ----------------
    # Convert preprocessed data to DataFrames
    log_features = preprocessing.named_transformers_['log'].get_feature_names_out(
        ['total_rooms', 'total_bedrooms', 'population', 'households']
    ).tolist()
    one_hot_features = preprocessing.named_transformers_['one_hot'].get_feature_names_out(
        ["ocean_proximity"]
    ).tolist()
    remainder_features = preprocessing.named_transformers_['remainder'].get_feature_names_out().tolist()

    feature_names = log_features + one_hot_features + remainder_features

    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=X_test.index)

    # Save preprocessed data as CSV
    X_train_preprocessed_df.to_csv(f'{output_folder}/X_train_preprocessed.csv', index=False)
    X_test_preprocessed_df.to_csv(f'{output_folder}/X_test_preprocessed.csv', index=False)
    Y_train.to_csv(f'{output_folder}/Y_train.csv', index=False)
    Y_test.to_csv(f'{output_folder}/Y_test.csv', index=False)

    print("Data preprocessing completed and saved as CSV files.")

if __name__ == "__main__":
    main_data_path = 'data/external/housing.csv'
    output_dir = 'data/processed'
    preprocess_data(main_data_path, output_dir)
