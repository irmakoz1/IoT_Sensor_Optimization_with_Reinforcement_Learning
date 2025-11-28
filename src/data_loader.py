import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.train_data = None
        self.val_data = None

    def load_data(self):
        data = pd.read_csv(self.filepath)
        data = data.dropna()
        self.data = data
        return self.data

    def encode_categories(self):
        self.data['scale_type'] = self.data['scale_type'].astype('category').cat.codes
        self.data['box_type'] = self.data['box_type'].astype('category').cat.codes
        self.data['initiatortype'] = self.data['initiatortype'].astype('category').cat.codes
        self.data['weightclassification'] = self.data['weightclassification'].map({
            'TOO_MANY': -1, 'TOO_FEW': -5, 'OK': 3
        })
        return self.data

    def train_val_split(self, test_size=0.2, random_state=42):
        self.train_data, self.val_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return self.train_data, self.val_data
