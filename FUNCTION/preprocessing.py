# IMPORT LIBRARY
import pandas as pd
import numpy as np
from joblib import dump

# Library for preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# FUNCTION TO PREPROCESS DATA
class DataPreprocessor:
    def __init__(self, target_column, save_path, file_path, data_train, data_test):
        self.target_column = target_column # Target variable
        self.save_path = save_path # Path to save the preprocessing pipeline
        self.file_path = file_path # Path to save feature names
        self.data_train = data_train
        self.data_test = data_test
    
    # Method to identify feature types
    def identify_feature_types(self):
        numerik = self.data_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        kategorikal = self.data_train.select_dtypes(include=['object']).columns.tolist()

        if self.target_column in numerik:
            numerik.remove(self.target_column)
        if self.target_column in kategorikal:
            kategorikal.remove(self.target_column)

        return numerik, kategorikal
    
    # Method to create preprocessing pipelines
    def create_pipelines(self, numerik):
        one_hot_fitur = ['DP', 'Varietas', 'Rayon', 'Bulan Tanam']
        ordinal_fitur = ['Tingkat Tanam']

        onehot_transform = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        ordinal_transform = Pipeline([
            ('encoder', OrdinalEncoder(categories=[['PC', 'R1', 'R2', 'R3']], handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        numeric_fitur = Pipeline([
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_fitur, numerik),
            ('onehot', onehot_transform, one_hot_fitur),
            ('ordinal', ordinal_transform, ordinal_fitur)
        ])
        return preprocessor
    
    # Method to final preprocess data
    def preprocess(self):
        numerik, kategorikal = self.identify_feature_types()
        preprocessor = self.create_pipelines(numerik)

        X_train = self.data_train.drop(columns=[self.target_column])
        y_train = self.data_train[self.target_column]
        X_test = self.data_test.drop(columns=[self.target_column])
        y_test = self.data_test[self.target_column]

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        dump(preprocessor, self.save_path)
        feature_names = preprocessor.get_feature_names_out()
        pd.DataFrame(columns=feature_names).to_csv(self.file_path, index=False)

        return X_train_processed, X_test_processed, y_train, y_test 
    
# example usage
# preprocessor = DataPreprocessor(target_column='target', save_path='preprocessor.joblib', file_path='feature_names.csv', data_train=train_df, data_test=test_df)
# X_train_processed, X_test_processed, y_train, y_test = preprocessor.preprocess()
# Alternative function-based approach
