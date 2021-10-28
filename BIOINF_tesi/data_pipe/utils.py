import pandas as pd
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
from sklearn.metrics import make_scorer
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr, kruskal, ranksums
import miceforest as mf
from imblearn.over_sampling import SMOTE


TYPE_TEST = ['wilcoxon_test','kruskal_wallis_test']
TYPE_AUGM_GENFEATURES = ['smote', 'double']


def MICE(X, random_state=100, verbose=False):
    """MICEforest is an iterative algorithm used for imputating missing data.
    Returns a single imputated dataset.

     Parameters
    ----------------
    X (pd.DataFrame): data.

    Returns
    ------------
    Imputated dataset.
    """

    # create object.
    kds = mf.KernelDataSet(
        X,
        mean_match_candidates=10,
        save_all_iterations=False,
        random_state=random_state)
    # run algorithm for 6 iterations and use all the processors.
    kds.mice(6, n_jobs=-1)
    if verbose:
        print(kds)
    # Return the imputated dataset.
    return kds.complete_data()



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
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)

    pos_index = y[y==1].index
    neg_index = y[y==0].index
        
    for col in X.columns: 
        pos_samples = X[col].reindex(index = pos_index)
        neg_samples = X[col].reindex(index = neg_index)
            
        _, p_value = kruskal(pos_samples, neg_samples)
        if p_value > kruskal_pval_threshold:
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
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)
    
    pos_index = y[y==1].index
    neg_index = y[y==0].index
        
    for col in X.columns:
        pos_samples = X[col].reindex(index = pos_index)
        neg_samples = X[col].reindex(index = neg_index)
            
        _, p_value = ranksums(pos_samples, neg_samples)
        if p_value > wilcoxon_pval_threshold:
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
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)
    
    pos_index = y[y==1].index
    neg_index = y[y==0].index
    pos_samples = X.reindex(index = pos_index).values
    neg_samples = X.reindex(index = neg_index).values
    
    _, p_value = kruskal(pos_samples, neg_samples)
    
    return(p_value)



def kruskal_wallis_test_pval(X,y):
    """Returns p-value of kruskal-wallis test between classes of X.
    
    Parameters
    ----------------
    X (pd.DataFrame): data.
    y (pd.Series): label.
    """
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)
    
    pos_index = y[y==1].index
    neg_index = y[y==0].index
    pos_samples = X.reindex(index = pos_index).values
    neg_samples = X.reindex(index = neg_index).values
    
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
            
            if abs(corr) >= spearman_corr_threshold:
                correlated[corr]=[col1,col2]
                
                if verbose:
                    print(f'correlated columns: {col1} - {col2}, Spearman Correlation {round(corr,4)}')
        # order by descending correlation
        ord_correlated = OrderedDict(sorted(correlated.items(), reverse=True))
        
        # return list of correlated pairs in descending correlation order
        return list(ord_correlated.values())
    
    

def remove_correlated_features(X, y, correlated_pairs, type_test='wilcoxon_test', verbose=False):
    """Removes the feature with the smallest effect on the target from a pair of highly correlated features.
    Effect on the target is calculated as the wilcoxon test p-value or the kruskal-wallis test p-value.
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
        # if none of the columns have been already eliminated compute their effect on the target.
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

            # drop the column of the pair with the smallest effect on the target.
            if pval_1 <= pval_2:
                X = X.drop([col2], axis=1)
            else:
                X = X.drop([col1], axis=1)
                    
    # return new dataframe with dropped correlated features.
    return X




def get_imbalance(y=None, n_pos=None, n_neg=None):
    """
    Returns percentage of class imbalance either by directly giving
    the labels or the number of positive and negative samples.

    Parameters
    ------------
    y (pd.Series): binary labels.
        Default: None
    n_pos (int): number of positive samples.
        Default: None
    n_neg (int): number of negative samples.
        Default: None

    Returns
    ------------
    tot.positive / tot.negative rounded to 2 decimals (float).
    """

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y=np.array(y)

    if isinstance(y, np.ndarray):
        n_pos = len(y[y==1])
        n_neg = len(y[y==0])

    return np.round(float(n_pos/n_neg),2)


def get_IR(y):
    """
    Returns imbalance ratio as tot.negative / tot.positive.

    Parameters
    ------------
    y (pd.Series): binary labels.
    """

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y=np.array(y.values)
    
    n_pos = len(y[y==1])
    n_neg = len(y[y==0])
    return float(n_neg/n_pos)



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


def double_rebalance(X, y, rebalance_threshold, imbalance, random_state): 
    """
    Rebalances data and labels by resampling of positive observations
    until the imbalance is equal to the rebalance_threshold. 

    Parameters
    --------------
    X (pd.DataFrame): genomic sequence data.
    y (pd.Series): binary labels.
    imbalance: current level of imbalance as ratio tot.positive/tot.negative.
    rebalance_threshold: desired level of imbalance as ratio tot.positive/tot.negative.

    Returns
    --------------
    pd.DataFrame, pd.Series
    """
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)

    # retrieve positive samples
    pos_index = y[y==1].index
    X_ = X.iloc[pos_index]
    X_.reset_index(drop=True, inplace=True)
    # create new labels with the same length as the positive samples.
    y_ = pd.Series([1]*len(pos_index))

    np.random.seed(random_state)
    # compute the number of positive observations needed to reach an imbalance = rebalance_threshold.
    n_obs = compute_rebalancing_obs(rebalance_threshold, y=y)
    # randomly draw positive samples.
    index = np.random.randint(0, len(X_), n_obs)
    # append the augmented positive to the original data.
    X = X.append(X_.iloc[index])
    y = y.append(y_.iloc[index])

    assert (len(X) == len(y))
    
    return X.reset_index(drop=True), y.reset_index(drop=True)



def reverse_strand_rebalance(X, y, rebalance_threshold, random_state):
    """
    Rebalances genomic sequence data and labels by adding complementary strands
    of positive observations until the imbalance between classes is equal to rebalance_threshold.

    Parameters
    --------------
    X (pd.DataFrame): data about genomic sequences.
    y (pd.Seres): binary labels.
    rebalance_threshold: desired level of imbalance as ratio tot.positive/tot.negative.
    random_state: initial seed.

    Returns
    --------------
    pd.DataFrame, pd.Series
    """
    
    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)
    
    # retrieve positive samples
    pos_index = y[y==1].index
    X_ = X.iloc[pos_index]
    # reverse genomic sequence
    X_ = X_.apply(lambda x: reverse_strand(x))
    X_.reset_index(drop=True, inplace=True)
    # create new labels with the same length as the positive samples.
    y_ = pd.Series([1]*len(pos_index))
    
    np.random.seed(random_state)
    # compute the number of positive observations needed to reach an imbalance = rebalance_threshold.
    n_obs = compute_rebalancing_obs(rebalance_threshold, y=y)
    # randomly draw positive observations
    index = np.random.randint(0, len(X_), n_obs)
    # append the augmented positive to the original data
    X = X.append(X_.iloc[index])
    y = y.append(y_.iloc[index])

    assert (len(X) == len(y))
    assert (get_imbalance(y) == rebalance_threshold)
    
    return X.reset_index(drop=True), y.reset_index(drop=True)



def reverse_strand_augment(X, y, rebalance_threshold=0.1, 
    random_state=123):
    """
    Augments data and labels by adding complementary strands. If rebalance=True, 
    it keeps an imbalance equal to rebalance_threshold, after having doubled the
    positive samples. Else, it doubles the whole dataset.

    Example 1:
    The initial imbalance is 0.06
    - rebalance_threshold = 0.1
    n.pos = 6 --> 12
    n.neg = 100 --> 120
    The new imbalance is 12/120 = 0.1.

    Example 2:
    The initial imbalance is 0.15
    - rebalance_threshold = 0.1
    n.pos = 15 --> 30
    n.neg = 100 --> 200
    The new imbalance is 30/200 = 0.15 > 0.1.


    Parameters
    --------------
    X (pd.DataFrame): data about genomic sequences.
    y (pd.Seres): binary labels.
    rebalance: whether to rebalance classes or not.
        Default: True
    rebalance_threshold: desired level of imbalance as ratio.
        Default: 0.1

    Returns
    --------------
    pd.DataFrame, pd.Series
    """

    if isinstance(y, pd.DataFrame):
        y=pd.Series(y.values)


    len_X_pre = len(X)
    # retrieve positive samples
    index_ = y[y==1].index
    X_pos = X.iloc[index_].copy()
    # reverse the strand
    X_pos = X_pos.apply(lambda x: reverse_strand(x))
    X_pos.reset_index(drop=True, inplace=True)
    # create new labels with the same length as the positive samples.
    y_pos = pd.Series([1]*len(index_))
    

    # retrieve negative samples
    index_ = y[y==0].index
    X_neg = X.iloc[index_].copy()
    # reverse the strand
    X_neg = X_neg.apply(lambda x: reverse_strand(x))
    X_neg.reset_index(drop=True, inplace=True)

    # calculate new imbalance after doubling the positive.
    y_ = y.append(y_pos)
    imbalance=get_imbalance(y_)
    # if the data were originally imbalanced, we cannot double all the negatives, but we need to
    #take a subsample of them so that the imbalance is equal to rebalance_threshold.
    if imbalance>rebalance_threshold: 
        # compute the number of positive observations needed to reach an imbalance = rebalance_threshold.
        n_obs = compute_rebalancing_obs(0.1, y=y_)
        # randomly draw positive observations
        np.random.seed(random_state)
        index = np.random.randint(0, len(X_neg), n_obs)
        # create new labels with the same length as the number of negative samples needed.
        y_neg = pd.Series([0]*n_obs)

        # append the augmented negative to the original data
        X=X.append(X_neg.iloc[index])
        y=y.append(y_neg)
        # append the augmented positive to the original data
        X=X.append(X_pos)
        y=y.append(y_pos)
        # NB: this is the correct order since when SMOTE augments data, first it appends
        # 0 then 1 to the original dataset
        assert (get_imbalance(y) == rebalance_threshold)

    else:
        # create new labels with the same length as the negative samples.
        y_neg = pd.Series([0]*len(index_))
        # append the augmented negative to the original data
        X=X.append(X_neg)
        y=y.append(y_neg)
        # append the augmented positive to the original data
        X=X.append(X_pos)
        y=y.append(y_pos)
        assert (len_X_pre*2 == len(X))

    assert (len(X) == len(y))
    
    return X.reset_index(drop=True), y.reset_index(drop=True)




def data_rebalancing(X, y, sequence=False, type_augm_genfeatures='smote', 
                        rebalance_threshold=0.1, random_state=123):
    """
    Performs data rebalancing through augmentation. The positive data points
    augmented to rebalance the dataset are appended to the original dataset.
    
    Parameters
    --------------
    X (pd.DataFrame): data.
    y (pd.Seres): labels.
    sequence (bool): if the data is a genomic sequence or not.
        Default: False
    type_augm_genfeatures: when the data are the epigenomic features,
        what kind of augmentation to apply. Possible choices are
        'smote' and 'double'.
        Default: 'smote'
    rebalance_threshold: desired level of imbalance as ratio.
        Default: 0.1
    random_state (int): initial random seed.
        Default: 123

    Returns
    --------------
    pd.DataFrame, pd.Series
    """
    
    TYPE_AUGM_GENFEATURES

    if type_augm_genfeatures not in TYPE_AUGM_GENFEATURES:
                raise ValueError(
                f"Argument 'type_augm_genfeatures' has an incorrect value: use one among {TYPE_AUGM_GENFEATURES}")


    imbalance = get_imbalance(y)
    # if the data are imbalanced rebalance them
    if imbalance < rebalance_threshold:
        
        if sequence: 
            X,y = reverse_strand_rebalance(X, y, rebalance_threshold, random_state)
            return X,y
        
        else:
            # oversampling using SMOTE by increasing the number of positive samples
            if type_augm_genfeatures == 'smote':
                oversample_SMOTE = SMOTE(k_neighbors=5, sampling_strategy = rebalance_threshold)
                X, y = oversample_SMOTE.fit_resample(X, y.ravel())
                return X.reset_index(drop=True), pd.Series(y)

            elif type_augm_genfeatures == 'double':
                X,y = double_rebalance(X, y, rebalance_threshold, imbalance, random_state)
                return X, y

    # if the data are not imbalanced, return the original data
    else:
        return X, y
    


def data_augmentation(X, y, sequence=False, 
                        rebalance_threshold=0.1, random_state=123):
    """
    Performs data augmentation and rebalancing if required.
    
    Parameters
    --------------
    X (pd.DataFrame): data.
    y (pd.Seres): labels.
    sequence (bool): if the data is a genomic sequence or not.
        Default: False
    type_augm_genfeatures: when the data are the epigenomic features,
        what kind of augmentation to apply. Possible choices are
        'smote' and 'double'.
        Default: 'smote'
    threshold: desired level of imbalance as ratio.
        Default: 0.15
    random_state (int): initial random seed.
        Default: 123

    Returns
    --------------
    pd.DataFrame, pd.Series
    """

    len_X_pre = len(X)
    imbalance = get_imbalance(y)

    # if data are imbalanced, augment and rebalance them, else just double the dataset.
    if sequence: 
                X,y = reverse_strand_augment(X, y, rebalance_threshold=rebalance_threshold, 
                    random_state=random_state)
                return X,y

    # if data are imbalanced, augment and rebalance them
    if imbalance < rebalance_threshold:
        n_pos=y[y==1].count()*2
        n_neg=y[y==0].count()

        sampling_strategy = {0: n_neg+compute_rebalancing_obs(0.1, n_pos=n_pos, n_neg=n_neg),
                                    1: n_pos}
        oversample_SMOTE = SMOTE(k_neighbors=5, sampling_strategy = sampling_strategy)
        X, y = oversample_SMOTE.fit_resample(X, y.ravel())

        assert( get_imbalance(y) == rebalance_threshold)
        return X.reset_index(drop=True), pd.Series(y)
        

    # if data are not imbalanced, just augment them by doubling the dataset
    else:
        n_pos=y[y==1].count()*2
        n_neg=y[y==0].count()*2

        sampling_strategy = {0: n_neg, 1: n_pos}
        oversample_SMOTE = SMOTE(k_neighbors=5, sampling_strategy = sampling_strategy)
        X, y = oversample_SMOTE.fit_resample(X, y.ravel())

        assert( len_X_pre*2 == len(X))
        return X.reset_index(drop=True), pd.Series(y)

        return X, y
    


def compute_rebalancing_obs(rebalance_threshold=0.1, y=None, n_pos=None, n_neg=None):
    """Computes the number of positive or negative observations needed to obtain an
    imbalance equal to rebalance_threshold, either from the labels or from the number
    of positive and negative samples.

    Parameters
    --------------
    y (pd.Series): binary labels.
        Default: None
    n_pos (int): number of positive samples.
        Default: None
    n_neg (int): number of negative samples.
        Default: None
    
    Returns
    --------------
    int.
    """
    if isinstance(y, np.ndarray):
        y=pd.Series(y)

    if isinstance(y, pd.Series):
        imbalance = get_imbalance(y)
        n_pos = y[y==1].count()
        n_neg = y[y==0].count()
    elif n_pos and n_neg:
        imbalance = get_imbalance(n_pos=n_pos, n_neg=n_neg)

    if imbalance > rebalance_threshold:
        return int((n_pos/rebalance_threshold) - n_neg)
    elif imbalance < rebalance_threshold: 
        return int((n_neg*rebalance_threshold) - n_pos)
    else:
        return 0
