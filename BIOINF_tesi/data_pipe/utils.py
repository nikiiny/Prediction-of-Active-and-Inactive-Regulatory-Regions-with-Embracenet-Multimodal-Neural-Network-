import pandas as pd
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import wilcoxon
from imblearn.over_sampling import SMOTE


TYPE_TEST = ['wilcoxon_test','kruskal_wallis_test']

    
def kruskal_wallis_test(X, y, kruskal_pval_threshold = 0.05,verbose=False):
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
        
    pos_index = y[y==1].index
    neg_index = y[y==0].index
        
    for col in X.columns:
        pos_samples = X[col][pos_index]
        neg_samples = X[col][neg_index]
            
        _, p_value = kruskal(pos_samples, neg_samples)
        if p_value < kruskal_pval_threshold:
            uncorrelated.add(col)
                
            if verbose:
                print(f'uncorrelated column: {col}, Kruskal-Wallis p-value: {p_value}')

    return uncorrelated
    
    
    
def wilcoxon_test(X, y, wilcoxon_pval_threshold = 0.05,verbose=False):
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
        
    pos_index = y[y==1].index
    neg_index = y[y==0].index
        
    for col in X.columns:
        pos_samples = X[col][pos_index]
        neg_samples = X[col][neg_index]
            
        _, p_value = wilcoxon(pos_samples, neg_samples)
        if p_value < wilcoxon_pval_threshold:
            uncorrelated.add(col)
                
            if verbose:
                print(f'uncorrelated column: {col}, Wilcoxon p-value: {p_value}')
            
    return uncorrelated



def wilcoxon_test_pval(X,y):
    """Returns p-value of wilcoxon test between classes of X.
    
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    """
    
    pos_index = y[y==1].index
    neg_index = y[y==0].index
    pos_samples = X[pos_index].values
    neg_samples = X[neg_index].values
    
    _, p_value = kruskal(pos_samples, neg_samples)
    
    return(p_value)



def kruskal_wallis_test_pval(X,y):
    """Returns p-value of kruskal-wallis test between classes of X.
    
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    """
    
    pos_index = y[y==1].index
    neg_index = y[y==0].index
    pos_samples = X[pos_index].values
    neg_samples = X[neg_index].values
    
    _, p_value = kruskal(pos_samples, neg_samples)
    
    return(p_value)




def spearman_corr(X, spearman_corr_threshold=0.75, verbose=False):
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
                    print(f'correlated columns: {col1} - {col2}, Spearman Correlation {round(corr,4)}')
        # order by descending correlation
        ord_correlated = OrderedDict(sorted(correlated.items(), reverse=True))
        
        # return list of correlated pairs in descending correlation order
        return list(ord_correlated.values())
    
    

def remove_correlated_features(X, y, correlated_pairs, type_test='wilcoxon_test', verbose=False):
    """Removes the less correlated feature with the target from a pair of highly correlated features.
    Correlation with the target is calculated as the wilcoxon test p-value or the kruskal-wallis test p-value.
    A lower p-value means a lower support to null hypothesis, so a stronger ability of the variable
    to discriminate between the different classes of the target.

    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label
    correlated_pairs (list): list of pairs of correlated features
    type_test (str): type of test to decide which column to remove from a pair.
        Possible values are: wilcoxon_test, kruskal_wallis_test.
        Default: wilcoxon_test
    verbose (bool): returns scores.
        Default: False

    Returns
    ----------------
    Dataframe with no correlated pairs.
    """

    for pair in correlated_pairs:
        col1, col2 = pair
        if col1 in X and col2 in X:
            x1 = X[col1]
            x2 = X[col2]
            
            if type_test not in TYPE_TEST:
                raise ValueError(
                f"Argument 'type_test' has an incorrect value: use one among {TYPE_TEST}")
            
            
            if type_test == 'wilcoxon_test':
                pval_1 = wilcoxon_test_pval(x1, y)
                pval_2 = wilcoxon_test_pval(x2, y)
            elif type_test == 'kruskal_wallis_test':
                pval_1 = kruskal_wallis_test_pval(x1, y)
                pval_2 = kruskal_wallis_test_pval(x2, y)

            if verbose:
                print(f'columns to compare: {col1} vs {col2}, p-values: {pval_1} vs {pval_2}')

            if pval_1 <= pval_2:
                X = X.drop([col2], axis=1)
            else:
                X = X.drop([col1], axis=1)
                    
    # return new dataframe with dropped correlated features.
    return X




def get_imbalance(y):
    """
    Returns percentage of class imbalance.

    Parameters
    ------------
    y (pd.Series): binary labels.
    """
    n_pos = y[y==1].count()
    n_neg = y[y==0].count()
    return float(n_pos/n_neg)



def reverse_strand(sequence):
    """
    Returns str of complementary strand.

    Parameters
    --------------
    sequence (str): genomic sequence (a,c,t,g).
    """
    nucleotides_dict = {'a':'t', 't':'a', 'c':'g', 'g':'c', 'n':'n'}
    sequence = list(sequence.lower())
    sequence = [nucleotides_dict[base] for base in sequence]
    
    return ''.join(sequence)



def reverse_strand_augment(X,y):
    """
    Augment data and labels by adding complementary strands.

    Parameters
    --------------
    X (pd.DataFrame): data about genomic sequences.
    y (pd.Seres): binary labels.
    """
    
    pos_index = y[y==1].index
    pos_X = X.iloc[pos_index]
            
    X = pos_X.apply(lambda x: reverse_strand(x))
    y = pd.Series([1]*len(pos_index))
            
    assert (len(X) == len(y))
    
    return X,y



def data_augmentation(X, y, sequence, threshold=0.15):
    """
    Performs data augmentation. ........ comment
    
    Attributes:
    X (pd.Series):
    y (pd.Series)
    """
    
    imbalance = get_imbalance(y)
    if imbalance <= threshold:
        
        if sequence:
            X_aug, y_aug = reverse_strand_augment(X,y)
            
            index = np.random.randint(0, len(X_aug), int( len(X_aug)*(threshold-imbalance)/imbalance ))
            
            X = X.append(X_aug.iloc[index])
            y = y.append(y_aug.iloc[index])
            
            return X.reset_index(drop=True), y.reset_index(drop=True)
        
        else:
            oversample_SMOTE = SMOTE(k_neighbors=3, sampling_strategy = threshold)
            X, y = oversample_SMOTE.fit_resample(X, y.ravel())
            
            return X.reset_index(drop=True), pd.Series(y)
    
    else:
        return X, y
    