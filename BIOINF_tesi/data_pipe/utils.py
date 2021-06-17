import pandas as pd
import numpy as np
import itertools
from scipy.stats import pointbiserialr
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import wilcoxon
from collections import defaultdict




def point_biserial_corr(X, y, pb_corr_threshold=0.05, verbose=False):
    """Point biserial correlation returns the correlation between a continuous and
    binary variable (target). It is a parametric test, so it assumes the data to be normally
    distributed.
        
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    pb_corr_threshold: threshold for eliminating X uncorrelated to y.
        Default: 0.05
    verbose (bool): returns scores.
        Default: False

    Returns
    ------------
    Set of columns uncorrelated with the target.
    """
        
    uncorrelated = set()

    for col in X.columns:
        x = X[col]
        corr,_ = pointbiserialr(x,y)
        if abs(corr) < pb_corr_threshold:
            uncorrelated.add(col)
            if verbose:
                print('uncorrelated column: {}, Point-biserial Correlation: {}'.format(col, round(corr,4)))
        
    return uncorrelated
    
    
def logistic_regression_corr(X, y, verbose=False):
    """The Logistic regression can be used to assess if a continuous variable has any 
    effect on the target. It doesn't assume anything about the distribution of the variables.
    The metric used is the area under the precision recall curve, which is compared to the 
    baseline (when the model guesses at random the target.

    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    verbose (bool): returns scores.
        Default: False

    Returns
    ----------------
    Set of columns uncorrelated with the target (AUPRC = baseline).
    """
        
    uncorrelated = set()
        
    for col in X.columns:
        x = X[col].values.reshape(-1, 1)

        # perform 3-folds cv with logistic regression
        cv = KFold(n_splits=3, random_state=123, shuffle=True)
        model = LogisticRegression()

        baseline_binary_pos = len(y[y==1]) / len(y)

        # compute the AUPRC for the positive score
        AUPRC = make_scorer(average_precision_score, average='weighted')
        scores = cross_val_score(model, x, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")
        
        if scores.mean() <= baseline_binary_pos:
            uncorrelated.add(col)
                
            if verbose:
                print('uncorrelated column: {}, AUPRC: {}, Baseline positive class: {}'.format(col, round(scores.mean(),4), round(baseline_binary_pos,4)))
        
    return uncorrelated

    
def kruskal_wallis_corr(X, y, kruskal_pval_threshold = 0.05,verbose=False):
    """The Kruskal–Wallis test by ranks, or one-way ANOVA on ranks, is a non-parametric method 
    for testing whether samples originate from the same distribution. The samples may have different
    sizes.
    The null hypothesis is that the median of all groups is equal, while the alternative hyphothesis is that the median
    of at least one group is different from the median of at least another group.
    In this case it is used to compare the distributions between positive and negative
        
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    kruskal_pval_threshold: p-value threshold for eliminating X uncorrelated to y.
        Default: 0.05
    verbose (bool): returns scores.
        Default: False

    Returns
    ------------
    Set of columns uncorrelated with the target.
    """
        
    uncorrelated = set()
        
    pos_index = y[y==1]
    neg_index = y[y==0]
        
    for col in X.columns:
        pos_samples = X[col][pos_index]
        neg_samples = X[col][neg_index]
            
        _, p_value = kruskal(pos_samples, neg_samples)
        if p_value < kruskal_pval_threshold:
            uncorrelated.add(col)
                
            if verbose:
                print('uncorrelated column: {}, Kruskal-Wallis p-value: {}'.format(col, p_value))

    return uncorrelated
    
    
    
def wilcoxon_corr(X, y, wilcoxon_pval_threshold = 0.05,verbose=False):
    """The Wilcoxon signed-rank test tests the null hypothesis that two related paired 
    samples come from the same distribution. In particular, it tests whether the distribution 
    of the differences x - y is symmetric about zero. It is a non-parametric version of the 
    paired T-test.
        
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    wilcoxon_pval_threshold: p-value threshold for eliminating X uncorrelated to y.
        Default: 0.05
    verbose (bool): returns scores.
        Default: False

    Returns
    ------------
    Set of columns uncorrelated with the target.
    """
        
    uncorrelated = set()
        
    pos_index = y[y==1]
    neg_index = y[y==0]
        
    for col in X.columns:
        pos_samples = X[col][pos_index]
        neg_samples = X[col][neg_index]
            
        _, p_value = wilcoxon(pos_samples, neg_samples)
        if p_value < wilcoxon_pval_threshold:
            uncorrelated.add(col)
                
            if verbose:
                print('uncorrelated column: {}, Wilcoxon p-value: {}'.format(col, p_value))
            
    return uncorrelated





def spearman_corr(X, spearman_corr_threshold=0.85, verbose=False):
        """Spearman correlation checks for linear correlation between continuous features.
        It is non-parametric, so normality of the variables is not necessary.

        Parameters
        ----------------
        X (pd.DataFrame): data.
        verbose (bool): returns scores.
            Default: False

        Returns
        ----------------
        List of pairs of highly correlated features (>0.85) in descending correlation order.
        """
        
        correlated = dict()
        
        for col1, col2 in itertools.combinations(X.columns, 2):
            corr, _ = spearmanr(X[col1].values, X[col2].values)
            
            if corr >= spearman_corr_threshold:
                correlated[corr]=[col1,col2]
                
                if verbose:
                    print('correlated columns: {} - {}, Spearman Correlation {}'.format(col1, col2, round(corr,4)))
        # order by descending correlation
        ord_correlated = OrderedDict(sorted(correlated.items(), reverse=True))
        
        # return list of correlated pairs in descending correlation order
        return list(ord_correlated.values())
    
    

def remove_correlated_features(X, y, correlated_pairs, verbose=False):
    """Removes the less correlated feature with the target from a pair of highly correlated features.
    Correlation with the target is calculated as the AUPRC of a logistic regression between
    the feature and the target.

    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label
    correlated_pairs (list): list of pairs of correlated features
    verbose (bool): returns scores.
        Default: False

    Returns
    ----------------
    Dataframe with no correlated pairs.
    """

    for pair in correlated_pairs:
        col1, col2 = pair
        if col1 in X and col2 in X:
            x1 = X[col1].values.reshape(-1, 1)
            x2 = X[col2].values.reshape(-1, 1)

            # perform 3-folds cv with logistic regression
            cv = KFold(n_splits=3, random_state=123, shuffle=True)
            model = LogisticRegression()

            # compute the AUPRC for the positive score
            AUPRC = make_scorer(average_precision_score, average='weighted')
            scores1 = cross_val_score(model, x1, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")
            scores2 = cross_val_score(model, x2, y, scoring=AUPRC, cv=cv, n_jobs=-1, error_score="raise")

            if verbose:
                print('columns to compare: {} vs {}, AUPRC: {} vs {}'.format(col1, col2, scores1.mean(), scores2.mean()))

            if scores1.mean() >= scores2.mean():
                X = X.drop([col2], axis=1)
                print('removed column: {}'.format(col2))
            else:
                X = X.drop([col1], axis=1)
                print('removed column:  {}'.format(col1))
                    
    # return new dataframe with dropped correlated features.
    return X