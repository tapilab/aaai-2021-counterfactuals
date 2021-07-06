The source and generated data files used in the experiment. Some files are too large to upload to the github repository, please contact the author to get files missing from this repository.

```
class Counterfactual:
    def __init__(self, df_train, df_test, moniker):
        display(df_train.head(1))
        self.moniker = moniker
        self.train = df_train
        self.test = df_test
        
def load_data(moniker):
    data_path = '/data/zwang/2020_S/Attention/Counterfactual/'   
    if(moniker == 'kindle'):
        ds = pickle.load(open(data_path+"kindle_ct/causal_sents/V_0906/ds_kindle.pkl", "rb"))
    elif(moniker == 'imdb'):
        ds = pickle.load(open(data_path+"imdb_ct/sentiment/combined/paired/paragraph/V_0906/ds_imdb.pkl", "rb"))
    elif(moniker == 'imdb_sents'):
        ds = pickle.load(open(data_path+"imdb_ct/sentiment/combined/paired/split_sents/V_0906/ds_imdb.pkl", "rb"))
    
    return ds
```

The **data structure (ds)** contains the following fields ```ds.__dict__.keys()```:

    - ds.moniker = 'kindle' | 'imdb' | 'imdb_sents' 
    - ds.train: data frame for training data, including generated counterfactuals using causal terms
    - ds.test: data frame for testing data
    - ds.identified_causal_terms: dataframe, causal terms identified using closest opposite matching score >= 0.95
    - ds.antonym_vocab: dataframe, the whole vocabulary and human annotated causal terms (causal terms are from train and test set)
    - ds.all_causal_terms: dataframe, the annotated causal terms in training set (a subset of antonym_vocab)
    - ds.top_terms: dataframe, terms with abs(coef) > threshold, with human annotated causal labels
