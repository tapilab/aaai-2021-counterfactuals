import random, pickle, re
import pandas as pd
import numpy as np
from itertools import product
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from PyDictionary import PyDictionary
dictionary=PyDictionary()

random.seed(42)

from Classification import fit_classifier, get_top_terms

class Counterfactual:
    def __init__(self, df_train, df_test, moniker):
        display(df_train.head(1))
        self.moniker = moniker
        self.train = df_train
        self.test = df_test



def get_antonyms(vocab, causal_words):
    """
    - antonyms: top term with opposite coefficient;
    - get antonyms for all words in the vocab
    - Help provide more options for manually edit counterfactual examples
    - # 90 min for imdb vocab
    """
    term_antonyms = {}
    for ti, term in enumerate(causal_words):
        try:
            term_coef = vocab[term]

            ant_terms = {} # antonym and its coef
            for ant in dictionary.antonym(term):
                if (ant in vocab) and (term_coef * vocab[ant] < 0): # opposite coef, 
                    ant_terms[ant] = vocab[ant]

            if(len(ant_terms) == 0):
                for syn in dictionary.synonym(term):
                    if(len(re.findall('\w+', syn)) == 1):
                        for ant in dictionary.antonym(syn):
                            if (ant in vocab) and (ant != term) and (term_coef * vocab[ant] < 0): # 
                                ant_terms[ant] = vocab[ant]
        except:
            continue
    
        term_antonyms[term] = ant_terms
#         term_antonyms.append({'term':term,'coef':term_coef, 'antonyms':ant_terms})
        
    return term_antonyms



def get_data(data_path,pre_process,moniker):
    """
    - get kindle or imdb from different files
    """
    if(moniker == 'kindle'):
        df_kindle = pickle.load(open(data_path+"kindle_ct/kindle_data.pkl",'rb'))
        df_train = df_kindle[df_kindle['flag']=='selected_train']
        df_test = df_kindle[df_kindle['flag']=='test']
        vocab_antonym = pd.read_csv(data_path+'kindle_ct/vocab_antonym_causal.csv')
    elif(moniker == 'imdb'):
        df_train = pre_process(data_file = data_path + "imdb_ct/sentiment/combined/paired/train_paired.tsv")
        df_test = pre_process(data_file = data_path + "imdb_ct/sentiment/combined/paired/test_paired.tsv")
        vocab_antonym = pd.read_csv(data_path+'imdb_ct/sentiment/imdb_vocab_antonym_causal.csv')
    elif(moniker == 'imdb_sents'):
        df_train = pickle.load(open(data_path+"imdb_ct/sentiment/combined/paired/split_sents/train_paired_sents.pkl", 'rb'))
        df_test = pickle.load(open(data_path+"imdb_ct/sentiment/combined/paired/split_sents/test_paired_sents.pkl", 'rb'))
        vocab_antonym = pd.read_csv(data_path+'imdb_ct/sentiment/imdb_vocab_antonym_causal.csv')
        
    return df_train, df_test, vocab_antonym

def get_causal_antonyms(data_path,pre_process,moniker):
    """
    1. Get train and test data from file and construct Counterfactual object
    2. Get top words
    3. Annotate causal words and generate antonyms for causal words (Get antonyms for full vocab in advance)
    """
    random.seed(42)
    
    # 1. Get train and test data from file and construct Counterfactual object
    df_train, df_test, vocab_antonym = get_data(data_path,pre_process,moniker)
        
    ds = Counterfactual(df_train, df_test, moniker)
        
    print('Train: %s' % str(Counter(df_train.label).items()))
    print('Test: %s' % str(Counter(df_test.label).items()))
        
    # 2. Get top words
    ds.full_vocab = vocab_antonym
    clf, vec = fit_classifier(train_text = df_train.text.values, train_label = df_train.label.values,
                                   test_text = df_test.text.values, test_label=df_test.label.values, report=True)
    
    ds.top_terms = get_top_terms(clf, vec, topn=0, min_coef=1.0, show_data=True)
    
    print('\n%d top terms: %d pos, %d neg\n' % (ds.top_terms.shape[0], 
                                                ds.top_terms[ds.top_terms.coef>0].shape[0],
                                                ds.top_terms[ds.top_terms.coef<0].shape[0]))
    
    # 3. Annotate causal words (load from pre-annotated file)
    ds.top_terms['causal'] = [ds.full_vocab[ds.full_vocab['term'] == item.term].causal.values[0] if item.term in ds.full_vocab.term.values else 0 for i, item in ds.top_terms.iterrows()]
    print('%d causal terms \n' % ds.top_terms[ds.top_terms['causal'] == 1].shape[0])
    
    # 4. Get antonyms for causal words    
    ds.top_terms['antonyms'] = [eval(ds.full_vocab[ds.full_vocab['term'] == item.term].antonyms.values[0]) if item.term in ds.full_vocab.term.values else {} for i, item in ds.top_terms.iterrows()]
    ds.top_terms['n_antonyms'] = ds.top_terms['antonyms'].apply(lambda x: len(x))
    print('\nGet antonyms for %d causal terms\n' % ds.top_terms[(ds.top_terms['n_antonyms'] > 0) & (ds.top_terms['causal'] == 1)].shape[0])
    
    return ds


def identify_causal_words(df, df_causal_terms, flag='causal', show_data=True):
    """
    Identify causal words in each sentence
    - Use CSR matrix from CountVectorizer instead of regular expression
    - flag = 'causal' or flag = 'bad' or flag='top'
    """
#     causal_wds = df_top_terms[df_top_terms['causal'] == 1]['term'].values
#     bad_wds = top_term_df[top_term_df['causal'] == 0]['term'].values

    df[flag+'_wds'] = df['text'].apply(lambda x: [wd for wd in re.findall('\w+', x.lower()) if wd in df_causal_terms.term.values])
    df['n_'+flag+'_wds'] = df[flag+'_wds'].apply(lambda x: len(x))
    
    if(show_data):
        print("%d out of %d sentences include %s words" % (df[df['n_'+flag+'_wds']>0].shape[0], df.shape[0], flag))
    
    
def generate_ct_sentences(df, df_causal_terms, flag='causal'):
    """
    Generate counterfactual sentences for those contain causal words:
        - substitute all the causal words to antonyms;
        - antonyms: top term with opposite coefficient;
        - If no antonyms, keep the original causal word;
    """
    random.seed(42)
    
    all_ct_wds = []
    for ri, row in df.iterrows():
        if row['n_'+flag+'_wds'] > 0:
            words = re.findall('\w+', row.text.lower())
            new_wds = []
            ct_wds = []
            for wd in words:
#                 wd_coef = ds.vocab[ds.vocab['term']==wd].coef.values[0]
                if(wd in df_causal_terms.term.values):
                    # randomly select antonym that has equal coef with current word
#                         max_coef = -(wd_coef - 0.2)
#                         min_coef = -(wd_coef + 0.2)
#                         sub_w = list(set(top_term_df[(top_term_df['coef']>min_coef) & (top_term_df['coef']<max_coef)]['term'].values).intersection(set(causal_wds)))

#                     sub_w = list(ds.vocab[ds.vocab['term'] == wd].antonyms.values[0].keys())
                    sub_w = list(df_causal_terms[df_causal_terms['term'] == wd].antonyms.values[0].keys())

                    if(len(sub_w) == 1):
                        ct_wd = str(sub_w[0])
                    elif(len(sub_w) > 1):
                        ct_wd = str(random.sample(sub_w,1)[0])
                    else: # if no antonyms then remove current word
                        ct_wd = wd
                        # print(ri, wd_coef, sub_w)

                    new_wds.append(ct_wd)
                    ct_wds.append(ct_wd)
                else:
                    new_wds.append(wd)
                
            if(new_wds == words): # no antonym for the causal word
                all_ct_wds.append([])
                df.loc[ri, 'ct_text_'+flag] = ' '
            else:    
                all_ct_wds.append(ct_wds)
                df.loc[ri, 'ct_text_'+flag] = ' '.join(new_wds)
        else:
            all_ct_wds.append([])
            df.loc[ri, 'ct_text_'+flag] = ' '
        
        
    df['ct_'+flag+'_wds'] = all_ct_wds              
        



def run_counterfactual(data_path, ds, df_causal_terms, data, flag='causal', show_data=False):
    """
    generate counterfactual for train and test data
    """
    # 5. Automatically generate counterfactual samples for both training and testing data
#     print("Generate counterfactual sentences:")
    identify_causal_words(ds.train, df_causal_terms,flag,show_data=False)
    generate_ct_sentences(ds.train, df_causal_terms,flag)
    
    if(show_data):
        display(ds.train.head(2))
    
    identify_causal_words(ds.test, df_causal_terms,flag,show_data=False)
    generate_ct_sentences(ds.test, df_causal_terms,flag)
    
    if(data != 'imdb'): # run for kindle, imdb_sents dataset
        ds.train['ct_label'] = ds.train['label'].apply(lambda x: 0-x)
        ds.test['ct_label'] = ds.test['label'].apply(lambda x: 0-x)
    
        df_annotate_ct = pd.read_csv(data_path+'kindle_ct/kindle_ct_edit_500.csv')
#         ds.test['ct_text_amt'] = [df_annotate_ct[df_annotate_ct['id']==idx]['ct_text_amt'].values[0] for idx in ds.test.index.values]
    
    if(show_data):
        display(ds.test.head(2))



def percentage_of_causal_words(ds,train_text, train_label, test_text, test_label,topn,min_coef):
    """
    Percentage of causal words among top-n terms
    """
    clf,vec = fit_classifier(train_text, train_label, test_text, test_label, report=True)
    df_top_terms = get_top_terms(clf,vec,topn,min_coef,show_data=False)
    
    df_top_terms['causal'] = [ds.full_vocab[ds.full_vocab['term'] == item.term].causal.values[0] if item.term in ds.full_vocab.term.values else 0 for i, item in df_top_terms.iterrows()]
    n_term = df_top_terms.shape[0]
    n_causal = df_top_terms[df_top_terms.causal == 1].shape[0]
    p_causal = float("%.3f" % (n_causal / n_term))
    
    print("%d causal terms among %d top terms: %.3f" % (n_causal, n_term, p_causal))
    
    

