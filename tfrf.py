import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin

class TFRFVectorizerMulticlass(CountVectorizer, TransformerMixin):
    def __init__(self, labels=None, strategy='max', **kwargs):
        super().__init__(**kwargs)
        self.labels = labels
        self.strategy = strategy
        self.rf_weights_ = None
        self.feature_names_ = None
        self.tf_matrix_ = None

    def fit(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        self.tf_matrix_ = X  # <-- penting: simpan TF matrix
        self.feature_names_ = self.get_feature_names_out()

        if y is None and self.labels is None:
            raise ValueError("Labels (y) must be provided for TF-RF weighting.")

        if y is None:
            y = self.labels

        y = np.array(y) # <-- dijadikan menjadi variabel numpy
        classes = np.unique(y) # [0, 270, ..., 630]
        n_features = len(self.feature_names_) # <--- panjang/banyaknya term unik

        rf_matrix = np.zeros((len(classes), n_features))

        for i, c in enumerate(classes): # [0, 270, ..., 630]
            in_class = (y == c) # True / False --> [True, True, False, True, False, ...]
            out_class = ~in_class            # --> [False, False, True, False, True, ...]

            '''
            X = [[1,0,0,0]
                [1,0,0,1]]
            '''

            df_in = np.asarray((X[in_class] > 0).sum(axis=0)).ravel()       # --> [2, 0, 0, 1]
            df_out = np.asarray((X[out_class] > 0).sum(axis=0)).ravel()     # --> [0, 1, 1, 0]

            # rf_matrix[i] = np.log(1 + (df_in / (df_out + 1))) # <-- adusment
            rf_matrix[i] = np.log(2 + (df_in / np.maximum(1, df_out))) # <-- sesuai sama rumus

        if self.strategy == 'max':
            self.rf_weights_ = rf_matrix.max(axis=0) # [0.8287, 0.72763, 1, ]
        elif self.strategy == 'mean':
            self.rf_weights_ = rf_matrix.mean(axis=0)
        elif self.strategy == 'per_class':
            self.rf_weights_ = rf_matrix
        else:
            raise ValueError("Unknown strategy: choose from 'max', 'mean', or 'per_class'")

        return self

    def transform(self, raw_documents, target_class=None):
        X = super().transform(raw_documents)

        if self.rf_weights_ is None:
            raise ValueError("Call fit() before transform().")

        if self.strategy == 'per_class':
            if target_class is None:
                raise ValueError("You must specify target_class when strategy='per_class'")
            weights = self.rf_weights_[target_class]
        else:
            weights = self.rf_weights_

        return X.multiply(weights)

    def export_rf_csv(self, filepath='tfrf_output.csv'):
        if self.rf_weights_ is None or self.tf_matrix_ is None:
            raise ValueError("Call fit() before export.")

        if self.strategy == 'per_class':
            raise NotImplementedError("CSV export not supported for 'per_class' strategy.")

        tf_rf_matrix = self.tf_matrix_.multiply(self.rf_weights_)
        '''
        [[1,0,0,0]    [2][0][0][0]
        [1,0,0,0]]
        '''
        tf_rf_mean = np.asarray(tf_rf_matrix.mean(axis=0)).ravel()

        df = pd.DataFrame({
            'term': self.feature_names_,
            'rf': self.rf_weights_,
            'avg_tf_rf': tf_rf_mean
        })

        df.to_csv(filepath, index=False)
        print(f"CSV TF-RF berhasil disimpan di: {filepath}")

    def export_rf_csv_custom(self, filepath='tfrf_output_custom.csv', doc_ids=None):
        if self.rf_weights_ is None or self.tf_matrix_ is None:
            raise ValueError("Call fit() before export.")

        if self.strategy == 'per_class':
            raise NotImplementedError("CSV export not supported for 'per_class' strategy.")

        tf = self.tf_matrix_.toarray()  # shape: (n_docs, n_terms)
        tf_df = pd.DataFrame(tf.T, columns=doc_ids if doc_ids else [f"A{i+1}" for i in range(tf.shape[0])])
        tf_df.insert(0, "Kata", self.feature_names_)
        tf_df.insert(0, "No.", range(1, len(self.feature_names_) + 1))
        tf_df["B"] = (tf > 0).sum(axis=0)  # document frequency
        tf_df["RF"] = self.rf_weights_

        tf_df.to_csv(filepath, index=False)
        print(f"CSV custom TF-RF berhasil disimpan di: {filepath}")

