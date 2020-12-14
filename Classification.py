# Evaluate classification performance with counterfactually augmented data
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def fit_classifier(train_text, train_label, test_text, test_label, report=True, train='comb'):
    """
    Given training data and test data
    """
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    if(train == 'comb'):
        X = vec.fit_transform(list(train_text) + list(test_text))
        X_train = vec.transform(train_text)
        X_test = vec.transform(test_text)
    elif(train == 'train'):
        X_train = vec.fit_transform(list(train_text))
        X_test = vec.transform(test_text)
        
    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, train_label)
    
    if(report):
        print(classification_report(test_label, clf.predict(X_test)))
        return clf, vec
    else:
        result = classification_report(test_label, clf.predict(X_test), output_dict=True)
        return float('%.3f' % result['accuracy'])
    
    
def get_top_terms(clf, vec, topn=0, min_coef=0.5, show_data=False):
    """
    - fit classifier
    - Select features by: topn or min_coef
    """
    df_vocab = pd.DataFrame({'term':vec.get_feature_names(),'coef':[float("%.3f" % c) for c in clf.coef_[0]]})
    
    if(topn == 0 and min_coef == 0):
        return df_vocab
    
    if(min_coef>0 and topn==0):
        df_top_terms = df_vocab[(df_vocab['coef']>= min_coef) | (df_vocab['coef']<= 0-min_coef)]
    elif(topn>0 and min_coef==0):
        df_vocab['coef_abs'] = df_vocab['coef'].apply(lambda x: abs(x))
        df_top_terms = df_vocab.sort_values(by=['coef_abs'], ascending=False).head(topn)
        df_top_terms.drop(columns=['coef_abs'],inplace=True)
    
    
    if(show_data):
        df_pos_terms = df_top_terms[df_top_terms['coef']>0]
        df_neg_terms = df_top_terms[df_top_terms['coef']<0]
        print("Features correlated with pos class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_pos_terms.sort_values(by=['coef'], ascending=False).iterrows()])
        print("\nFeatures correlated with neg class: \n", [item['term']+'/'+str(item['coef']) for i, item in df_neg_terms.sort_values(by=['coef'], ascending=True).iterrows()])
    
    
    return df_top_terms



def do_cv(text, labels, display=True):
    """
    Classifier with different feature representation:
        - bag-of-words;
        - Bert embedding;
    Evaluate with 5-fold cross_validation;
    """
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    X = vec.fit_transform(text)

    y = np.array(labels)
    print(Counter(y))
    
    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    preds = np.zeros(len(y))
    
    for train, test in kf.split(X):
        clf.fit(X[train], y[train])
        preds[test] = clf.predict(X[test])
        
    if(display):
        print(classification_report(y, preds))
    else:
        result = classification_report(y, preds, output_dict=True)
        return result['accuracy']
    
    
def domain_transfer(df_source, df_target, vectorizer='count', n=5):
    """
    Classifier with different feature representation:
        - bag-of-words;
        
    Evaluate with 5-fold cross_validation;
    """
    
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    X = vec.fit_transform(list(df_source.text.values) + list(df_target.text.values))
    X_source = X[:df_source.shape[0]]
    X_target = X[df_source.shape[0]:]

    y_source = df_source.label.values
    y_target = df_target.label.values
    print(Counter(y_source), Counter(y_target))
    
    clf = LogisticRegression(class_weight='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_source, y_source)
    
    print(classification_report(y_target, clf.predict(X_target)))
        

def select_sents(df,keywords):
    """
    - Select sentences that contain keywords
    """
    vec = CountVectorizer(min_df=5, binary=True, max_df=.8)
    X = vec.fit_transform(df.text.values)
    y = df.label.values
    
    wd_sents = {}
    sent_idx = set()
    for wd in keywords:
        try:
            s_idx = np.nonzero(X[:,vec.vocabulary_[wd]])[0]
            wd_sents[wd] = s_idx
            sent_idx.update(s_idx)
        except:
            continue
    return wd_sents, sent_idx