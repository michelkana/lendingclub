


```
#RUN THIS CELL 
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
```





<style>
blockquote { background: #AEDE94; }
h1 { 
    padding-top: 25px;
    padding-bottom: 25px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
h2 { 
    padding-top: 10px;
    padding-bottom: 10px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}

div.exercise {
	background-color: #ffcccc;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
}
div.theme {
	background-color: #DDDDDD;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 18pt;
}
div.gc { 
	background-color: #AEDE94;
	border-color: #E9967A; 	 
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 12pt;
}
p.q1 { 
    padding-top: 5px;
    padding-bottom: 5px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}
header {
   padding-top: 35px;
    padding-bottom: 35px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
</style>





<hr style="height:2pt">



```
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.api import OLS

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import math
from scipy.special import gamma

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from IPython.display import display

import random


```


    /usr/local/lib/python3.6/dist-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    



```
!pip install --upgrade pip
!pip install -U sklearn
!pip install imblearn
!pip install textblob
```


    Requirement already up-to-date: pip in /usr/local/lib/python3.6/dist-packages (18.1)
    Collecting sklearn
      Downloading https://files.pythonhosted.org/packages/1e/7a/dbb3be0ce9bd5c8b7e3d87328e79063f8b263b2b1bfa4774cb1147bfcd3f/sklearn-0.0.tar.gz
    Requirement already satisfied, skipping upgrade: scikit-learn in /usr/local/lib/python3.6/dist-packages (from sklearn) (0.19.2)
    Building wheels for collected packages: sklearn
      Running setup.py bdist_wheel for sklearn ... [?25l- done
    [?25h  Stored in directory: /root/.cache/pip/wheels/76/03/bb/589d421d27431bcd2c6da284d5f2286c8e3b2ea3cf1594c074
    Successfully built sklearn
    Installing collected packages: sklearn
    Successfully installed sklearn-0.0
    Collecting imblearn
      Downloading https://files.pythonhosted.org/packages/81/a7/4179e6ebfd654bd0eac0b9c06125b8b4c96a9d0a8ff9e9507eb2a26d2d7e/imblearn-0.0-py2.py3-none-any.whl
    Collecting imbalanced-learn (from imblearn)
    [?25l  Downloading https://files.pythonhosted.org/packages/e5/4c/7557e1c2e791bd43878f8c82065bddc5798252084f26ef44527c02262af1/imbalanced_learn-0.4.3-py3-none-any.whl (166kB)
    [K    100% |████████████████████████████████| 174kB 6.5MB/s 
    [?25hRequirement already satisfied: scipy>=0.13.3 in /usr/local/lib/python3.6/dist-packages (from imbalanced-learn->imblearn) (1.1.0)
    Requirement already satisfied: numpy>=1.8.2 in /usr/local/lib/python3.6/dist-packages (from imbalanced-learn->imblearn) (1.14.6)
    Collecting scikit-learn>=0.20 (from imbalanced-learn->imblearn)
    [?25l  Downloading https://files.pythonhosted.org/packages/10/26/d04320c3edf2d59b1fcd0720b46753d4d603a76e68d8ad10a9b92ab06db2/scikit_learn-0.20.1-cp36-cp36m-manylinux1_x86_64.whl (5.4MB)
    [K    100% |████████████████████████████████| 5.4MB 5.3MB/s 
    [?25hInstalling collected packages: scikit-learn, imbalanced-learn, imblearn
      Found existing installation: scikit-learn 0.19.2
        Uninstalling scikit-learn-0.19.2:
          Successfully uninstalled scikit-learn-0.19.2
    Successfully installed imbalanced-learn-0.4.3 imblearn-0.0 scikit-learn-0.20.1
    Collecting textblob
    [?25l  Downloading https://files.pythonhosted.org/packages/7c/7d/ad09a26b63d4ad3f9395840c72c95f2fc9fa2b192094ef14e9e720be56f9/textblob-0.15.2-py2.py3-none-any.whl (636kB)
    [K    100% |████████████████████████████████| 645kB 5.9MB/s 
    [?25hRequirement already satisfied: nltk>=3.1 in /usr/local/lib/python3.6/dist-packages (from textblob) (3.2.5)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from nltk>=3.1->textblob) (1.11.0)
    Installing collected packages: textblob
    Successfully installed textblob-0.15.2
    



```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import FunctionTransformer 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

from sklearn.svm import SVC 
from xgboost.sklearn import XGBClassifier 
import itertools
from sklearn.neighbors import KNeighborsClassifier

#from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from textblob import Word
from sklearn.ensemble import VotingClassifier 


```



    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-5-5464ca314e22> in <module>()
          3 from sklearn.utils import resample
          4 from sklearn.tree import DecisionTreeClassifier
    ----> 5 from sklearn.ensemble import RandomForestClassifier
          6 from sklearn.ensemble import RandomForestRegressor
          7 from sklearn.ensemble import AdaBoostClassifier
    

    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/__init__.py in <module>()
          5 
          6 from .base import BaseEnsemble
    ----> 7 from .forest import RandomForestClassifier
          8 from .forest import RandomForestRegressor
          9 from .forest import RandomTreesEmbedding
    

    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/forest.py in <module>()
         61 from ..exceptions import DataConversionWarning, NotFittedError
         62 from .base import BaseEnsemble, _partition_estimators
    ---> 63 from ..utils.fixes import parallel_helper, _joblib_parallel_args
         64 from ..utils.multiclass import check_classification_targets
         65 from ..utils.validation import check_is_fitted
    

    ImportError: cannot import name '_joblib_parallel_args'

    

    ---------------------------------------------------------------------------
    NOTE: If your import is failing due to a missing package, you can
    manually install dependencies using either !pip or !apt.
    
    To view examples of installing some common dependencies, click the
    "Open Examples" button below.
    ---------------------------------------------------------------------------
    


# Prediction of Charge-Offs

We will consider the loan status as the response variable, a binary outcome for a loan with value 1 for fully paid and 0 for Charged Off.

We will work with data previously cleaned and augmented with census information. We will use a subset of loans which were fully paid or charged-off.



```
#df_loan_accepted_census_cleaned = pd.read_csv('df_loan_accepted_census_cleaned_closed_10.csv')
# 10% of all closed loans between 2007 and 2015, cleaned and augmented with census data - 73 MB
df_loan_accepted_census_cleaned = pd.read_csv('http://digintu.tech/tmp/cs109a/df_loan_accepted_census_cleaned_closed_2007-2015_10.csv') 
# 50% of all closed loans between 2007 and 2015, cleaned and augmented with census data - 364 MB
#df_loan_accepted_census_cleaned = pd.read_csv('http://digintu.tech/tmp/cs109a/df_loan_accepted_census_cleaned_closed_2007-2015_50.csv')
```




```
df_loan = df_loan_accepted_census_cleaned.copy()
df_loan = df_loan[df_loan.loan_status.isin(['Charged Off', 'Fully Paid'])]
```


## Features Selection

Our goal is now to do exploratory analysis using predictive models in order to find important features in closed loans.

Statistical tests can be used to select features that have the strongest relationship with the response variable. 

The Recursive Feature Elimination works by recursively removing variables and building a model on those variables that remain. Model accuracy is used to select the variables which contribute the most to the response.

In this section we use a model-based approach of features selection using bagged trees and PCA. 



### Manual features selection

The following list of predictors are those which we MUST not use since they are data gathered AFTER the loan is funded. The reason to exclude them is because these features will not be available in unseen future dataset. Those features are highly correlated to charged-off loans and would otherwise bias our results.



```
not_predictor = [
'chargeoff_within_12_mths',   
'collection_recovery_fee',
'debt_settlement_flag',
'debt_settlement_flag_date',
'deferral_term',
'funded_amnt_inv',
'funded_amnt',
'hardship_amount',
'hardship_dpd',
'hardship_end_date',
'hardship_flag',
'hardship_last_payment_amount',
'hardship_length',
'hardship_loan_status',
'hardship_payoff_balance_amount',
'hardship_reason',
'hardship_start_date',
'hardship_status',
'hardship_type',
'last_credit_pull_d',
'last_fico_range_high',
'last_fico_range_low',
'last_pymnt_amnt',
'last_pymnt_d',
'next_pymnt_d',
'orig_projected_additional_accrued_interest',
'out_prncp',
'out_prncp_inv',
'payment_plan_start_date',
'pymnt_plan',
'recoveries',
'settlement_amount',
'settlement_date',
'settlement_percentage',
'settlement_status',
'settlement_term',
'total_pymnt',
'total_pymnt_inv',
'total_rec_int',
'total_rec_late_fee',
'total_rec_prncp',
'verification_status'
]
```


We drop the index column.



```
not_predictor  += ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1']
```


We drop the id, URL



```
not_predictor  += ['id', 'url']
```


We drop the employment title, loan title and description, which contains too many distinct values and cannot be easily categorized.



```
not_predictor += ['emp_title', 'title', 'desc']
```


We drop the success flag since it contains the same information as the loan status.



```
not_predictor += ['success']
```


We remove following the issue date, quarter and year, which is less relevant for future loans, we only keep the issue month (Jan to Dec).



```
not_predictor += ['issue_q', 'issue_d']
```


The grade and subgrade are categories which the LendingClub match to interest rates. Although the categories are fixed, the interest rate can slightly change within each category over the time. 

The term is calculated using the amount and the interest rate.

It follows that `int_rate`, `grade`, `sub_grade` are correlated. Furthermore `loan_amnt`, `int_rate` and `term` define the `installment`. These relationships would bring collinearity into our model for features selection.

We will therefore work with `loan_amnt`, `sub_grade` and `term`. If it comes out that these features are important for predicting charge-off, we will conclude that their related variables `grade`, `int_rate`, and `installment` are also important.



```
not_predictor += ['grade', 'int_rate', 'installment']
```


As far as FICO is concerned, there are 6 variables. Four of them are determined at the initial loan application, thus we can use them. It doesn't seem that they are updated. 

These 2 are significant and collinear, so only one needs to be selected. We choose

    - fico_range_high
    - fico_range_low

These 2 are not so significant and, we believe are used for joint applications.

    - sec_app_fico_range_high
    - sec_app_fico_range_low


However, These two are created, and undoubtedly updated, throughout the loan life. They should not be used:

    - last_fico_range_high
    - last_fico_range_low



```
not_predictor += ['sec_app_fico_range_high', 'sec_app_fico_range_low', 'last_fico_range_high', 'last_fico_range_low']
```


**Correlation and redundancy**

Features-pairs which correlate by either -1 or +1 can be considered to be redundant. However High absolute correlation does not imply redundancy of features in the context of classification, see textbook Feature Extraction - Foundations and Applications by I. Guyon et al. (p.10, figure 2 (e)). Therefore we will have a closer look at each correlation.



```
#https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
def find_high_correlated_features(frame):
    new_corr = frame.corr()
    new_corr.loc[:,:] = np.tril(new_corr, k=-1) 
    new_corr = new_corr.stack()
    print(new_corr[(new_corr > 0.80) | (new_corr < -0.80)])
    
predictor = list(set(df_loan_accepted_census_cleaned.columns.values)-set(not_predictor))
find_high_correlated_features(df_loan_accepted_census_cleaned[predictor])   
```


    open_acc                    num_sats               0.901231
    num_op_rev_tl               num_sats               0.838576
                                num_rev_tl_bal_gt_0    0.848086
    fico_range_low              fico_range_high        1.000000
    male_pct                    female_pct            -1.000000
    num_actv_bc_tl              num_bc_sats            0.846392
                                num_rev_tl_bal_gt_0    0.831199
    avg_cur_bal                 tot_hi_cred_lim        0.807931
    tot_cur_bal                 tot_hi_cred_lim        0.977334
                                avg_cur_bal            0.843135
    total_bc_limit              bc_open_to_buy         0.839302
    total_il_high_credit_limit  total_bal_ex_mort      0.873770
    num_actv_rev_tl             num_rev_tl_bal_gt_0    0.988788
                                num_op_rev_tl          0.844565
                                num_actv_bc_tl         0.833955
    num_rev_accts               num_op_rev_tl          0.812922
    num_bc_tl                   num_rev_accts          0.870041
    dtype: float64
    



```
not_predictor += ['fico_range_low','male_pct','num_rev_tl_bal_gt_0', 'num_actv_rev_tl','open_il_12m','open_rv_12m','avg_cur_bal','tot_hi_cred_lim','num_bc_tl']
```


`fico_range_low` and `fico_range_low` are highly correlated. We keep the high value.

`male_pct` The probability for the borrower being male can be removed, since `female_pct` conveys that information as well.

The Number of revolving trades with balance >0 `num_rev_tl_bal_gt_0` and the Number of currently active revolving trades `num_actv_rev_tl` are highly correlated with the Number of currently active bankcard accounts `num_actv_bc_tl`. It is safe to remove the formers for our classification task for identifying fully paid loans from charged-off ones.

The number of open credit lines `open_acc` (preapproved loans between a financial institution and borrower that may be used repeatedly up to a certain limit and can subsequently be paid back prior to payments coming due) in the borrower's credit file is highly correlated with the Number of satisfactory accounts (good standing accounts that have been paid in full and on time) `num_sats`. We will keep both features.

`open_il_24m` and `open_il_12m` are the Number of installment accounts opened in past 24 and 12 months respectively. Both values are strongly correlated. We will consider 24 months period since it includes 12 months period. We will handle `open_rv_12m` and `open_rv_24m` in the same way (Number of revolving trades opened in past 12, resp. 24 months).

`tot_cur_bal`, `avg_cur_bal` Total and average current balance of all accounts are strongly related to `tot_hi_cred_lim`, Total high credit/credit limit. We will only keep the total current balance.

`num_rev_accts` Number of revolving accounts (account created by a lender to represent debts where the outstanding balance does not have to be paid in full every month by the borrower to the lender) and `num_bc_tl` Number of bankcard accounts are correlated. This is because credit cards are usually considered as revolving accounts. We assume that the number of revolving accounts better describe the risk for loans and we will remove the number of bankcard accounts.

`total_bal_ex_mort` Total credit balance excluding mortgage and `total_il_high_credit_limit` Total installment high credit/credit limit are correlated. With an installment account, the borrower pays back the loan plus interest by making regular payments for a fixed amount of time. We will keep both features.

`bc_open_to_buy` Total open to buy on revolving bankcards (credit cards) can be considered as a subset of `total_bc_limit` Total bankcard high credit/credit limit, but both information can differ in many situations. We keep both.

**Features values distribution**

For each remaining feature, we will plot the distribution of their values in both charged-off and fully paid categories. This will help us seing how they might impact the decision boundaries.



```
nb = 1
for var in [x for x in df_loan.columns.values if x not in not_predictor]:
    if df_loan[var].dtype == np.float64 or df_loan[var].dtype == np.int64:
        nb = nb + 1
fig, ax = plt.subplots(nb//2, 2, figsize=(15,90))
i = 0
for var in [x for x in df_loan.columns.values if x not in not_predictor]:
    if df_loan[var].dtype == np.float64 or df_loan[var].dtype == np.int64:
        sns.kdeplot(df_loan[df_loan.loan_status=='Charged Off'][var], label='Charged Off', ax=ax[i//2,i % 2])
        sns.kdeplot(df_loan[df_loan.loan_status!='Charged Off'][var], label='Fully Paid', ax=ax[i//2,i % 2])
        ax[i//2,i % 2].set_ylabel(var)
        i = i + 1
        
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/numpy/core/_methods.py:26: RuntimeWarning: invalid value encountered in reduce
      return umr_maximum(a, axis, None, out, keepdims)
    


![png](4%29%20Models_files/4%29%20Models_33_1.png)


Looking at the plots above we see that the distribution for census-related features is almost the same accross both classes of loans. We will investigate those features closer using mode-based features selection in the next section.

**One-hot encoding**

We turn the loan status into a binary variable



```
df_loan.replace({'loan_status':{'Charged Off': 0, 'Fully Paid': 1}}, inplace=True)
df_loan.loan_status = df_loan.loan_status.astype('int')
```


We convert the `earliest_cr_line` feature to the number of years between the earliest credit line and the year when the loan was requested.



```
df_loan.earliest_cr_line = pd.to_datetime(df_loan.issue_d, format='%b-%Y').dt.to_period('Y') - pd.to_datetime(df_loan.earliest_cr_line, format='%b-%Y').dt.to_period('Y')
df_loan.earliest_cr_line = df_loan.earliest_cr_line.astype('int')
```


We turn categorical features into binary variables.



```
df_loan.replace({'term':{36: 1, 60: 0}},inplace=True)
```




```
df_loan = pd.get_dummies(df_loan, columns=['sub_grade', 'emp_length', 'home_ownership',
                                          'purpose', 'issue_m', 'addr_state', 'zip_code','earliest_cr_line'], drop_first=True)
```


**Remove irrelevant features**

Let's remove all indentified features above.



```
df_loan.drop(columns=list(set(not_predictor) & set(df_loan.columns.values)), inplace=True)
```


### Imbalanced Dataset



```
As we see below, the data is unbalanced, with Fully Paid loans being the majority class.
```




```
df_loan.loan_status.value_counts()
```





    1    633159
    0    146801
    Name: loan_status, dtype: int64



`X` is the feature matrix. `Y` is the response vector.



```
X, Y = df_loan[df_loan.columns.difference(['loan_status'])], df_loan['loan_status']
```


We choose to split the whole dataset to 90% training, 10% test.



```
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=9001)
```


Let's further split the training set into a 80% training and a 20% validation set.



```
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=9001)
```


Let's reduce the dimension of a subset of the data using Principal Component Analysis (PCA) and display the imbalanced classes in a 2D plot.



```
X_train_subset = X_train[:100]
y_train_subset = y_train[:100]
pca = PCA(n_components=2)
X_train_subset = pca.fit_transform(X_train_subset)
```




```
y_train_subset.value_counts()
```





    1    77
    0    23
    Name: loan_status, dtype: int64





```
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    labels = ['Charged Off', 'Fully Paid']
    for d, l, c, m in zip(labels, np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=d, marker=m, alpha=.5
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.show()
```




```
plot_2d_space(X_train_subset, y_train_subset, 'Imbalanced dataset')
```



![png](4%29%20Models_files/4%29%20Models_58_0.png)


A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of either removing samples from the majority class (under-sampling) or adding more examples from the minority class (over-sampling). Both strategies can also be applied at the same time.

As shown below with under-sampling, we tend to loose valuable information, which can increase bias.



```
rus = RandomUnderSampler(return_indices=True)
X_train_subset_rus, y_train_subset_rus, id_rus = rus.fit_sample(X_train_subset, y_train_subset)
print(X_train_subset.shape[0] - X_train_subset_rus.shape[0], 'random samples removed')
plot_2d_space(X_train_subset_rus, y_train_subset_rus, 'Random under-sampling')
```



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-40-96181afb096a> in <module>()
    ----> 1 rus = RandomUnderSampler(return_indices=True)
          2 X_train_subset_rus, y_train_subset_rus, id_rus = rus.fit_sample(X_train_subset, y_train_subset)
          3 print(X_train_subset.shape[0] - X_train_subset_rus.shape[0], 'random samples removed')
          4 plot_2d_space(X_train_subset_rus, y_train_subset_rus, 'Random under-sampling')
    

    NameError: name 'RandomUnderSampler' is not defined


In over-sampling the most naive strategy is to generate new samples by randomly sampling with replacement the current available samples, which can cause overfitting.



```
ros = RandomOverSampler()
X_train_subset_ros, y_train_subset_ros = ros.fit_sample(X_train_subset, y_train_subset)
print(X_train_subset_ros.shape[0] - X_train_subset.shape[0], 'random samples added')
plot_2d_space(X_train_subset_ros, y_train_subset_ros, 'Random over-sampling')
```


    54 random samples added
    


![png](4%29%20Models_files/4%29%20Models_62_1.png)


A number of more sophisticated resapling techniques have been proposed in the scientific literature, especially using the Python library imbalanced-learn (https://imbalanced-learn.org). SMOTE (Synthetic Minority Oversampling TEchnique) consists of creating new samples for the minority class, by picking a sample from that class and computing the k-nearest neighbors, then adding a new sample between the chosen sample and its neighbors.



```
smote = SMOTE(ratio='minority')
X_train_subset_sm, y_train_subset_sm = smote.fit_sample(X_train_subset, y_train_subset)

plot_2d_space(X_train_subset_sm, y_train_subset_sm, 'SMOTE over-sampling')
```



![png](4%29%20Models_files/4%29%20Models_64_0.png)


We will use SMOTE to balance our training dataset.



```
X_train, y_train = smote.fit_sample(X_train, y_train)
```




```
print('The Charged-Off to Fully Paid ratio in the balanced training set is now: ', len(y_train[y_train==0])/len(y_train[y_train==1]))
```


    The Charged-Off to Fully Paid ratio in the balanced training set is now:  1.0
    



```
#np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_train_upsampled_smote.csv', 
#           X_train, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
```




```
#np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_train_upsampled_smote.csv', 
#           y_train, fmt='%.2f', delimiter=',', header='loan_status')
```


### Model-based features selection

We will use classifiers on the dataset in order to get a better understanding on how features are related to loan outcome as fully paid or unpaid.

The function below fit a random forest to the training set and display the classification accuracy on test.



```
def run_random_forest(X_train, y_train, X_val, y_val, size, depth):
    randomf = RandomForestClassifier(n_estimators=size, max_depth=depth).fit(X_train, y_train)
    accuracy_train = randomf.score(X_train, y_train)
    accuracy_val = np.mean(cross_val_score(randomf, X_val, y_val, cv=5, scoring='roc_auc'))
    print('RANDOM FOREST')
    print('Number of trees: ', size)
    print('Tree depth: ', depth)
    print('Accuracy, Training Set: {0:0.2%}'.format(accuracy_train))
    print('CV Accuracy, Val Set: {0:0.2%}'.format(accuracy_val))
    return accuracy_train, accuracy_val, randomf
```


We can now see how a forest of 100 depth-20 trees performs on our data.



```
rf_accuracy_train, rf_accuracy_val, randomf = run_random_forest(X_train, y_train, X_val, y_val, 100, 20)
```


    RANDOM FOREST
    Number of trees:  100
    Tree depth:  20
    Accuracy, Training Set: 94.77%
    CV Accuracy, Val Set: 69.57%
    



```
print('CV Accuracy, Test Set: {0:0.2%}'.format(np.mean(cross_val_score(randomf, X_test, y_test, cv=5, scoring='roc_auc'))))
```


    CV Accuracy, Test Set: 69.94%
    

Below we have a ranking of features as computed by our random forest, depending on their Gini importance in the prediction of loan outcome.



```
feature_importances = pd.DataFrame(randomf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False).reset_index().rename(columns={'index':'feature'})
feature_importances.head(10)
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>term</td>
      <td>0.072537</td>
    </tr>
    <tr>
      <th>1</th>
      <td>purpose_debt_consolidation</td>
      <td>0.050164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>inq_last_6mths</td>
      <td>0.037292</td>
    </tr>
    <tr>
      <th>3</th>
      <td>emp_length_10</td>
      <td>0.029273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>acc_open_past_24mths</td>
      <td>0.023912</td>
    </tr>
    <tr>
      <th>5</th>
      <td>num_tl_op_past_12m</td>
      <td>0.023211</td>
    </tr>
    <tr>
      <th>6</th>
      <td>purpose_credit_card</td>
      <td>0.023188</td>
    </tr>
    <tr>
      <th>7</th>
      <td>percent_bc_gt_75</td>
      <td>0.018610</td>
    </tr>
    <tr>
      <th>8</th>
      <td>fico_range_high</td>
      <td>0.017118</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mths_since_recent_inq</td>
      <td>0.016534</td>
    </tr>
  </tbody>
</table>
</div>



**Important Features**

We will now use SKLearn meta-transformet SelectFromModel to discard irrelevant features using the features importance produced by our random forest.



```
fs_model = SelectFromModel(randomf, prefit=True, threshold=0.004)
outcome = fs_model.get_support()
features_list_orig = X.columns.values
features_list_new = []
for i in range(0,len(features_list_orig)):
    if outcome[i]:
        features_list_new.append(features_list_orig[i])
print('{} features were selected from the {} original hot-encoded ones'.format(len(features_list_new), len(features_list_orig)))
print(features_list_new)
```


    86 features were selected from the 1091 original hot-encoded ones
    ['Asian_pct', 'Black_pct', 'Graduate_Degree_pct', 'Hispanic_pct', 'Native_pct', 'Population', 'White_pct', 'acc_open_past_24mths', 'addr_state_CA', 'addr_state_TX', 'annual_inc', 'bc_open_to_buy', 'bc_util', 'delinq_2yrs', 'dti', 'earliest_cr_line_12', 'earliest_cr_line_14', 'emp_length_1', 'emp_length_10', 'emp_length_2', 'emp_length_3', 'emp_length_5', 'employment_2016_rate', 'female_pct', 'fico_range_high', 'home_ownership_OWN', 'home_ownership_RENT', 'household_family_pct', 'inq_last_6mths', 'issue_m_Aug', 'issue_m_Dec', 'issue_m_Jan', 'issue_m_Jul', 'issue_m_Jun', 'issue_m_Mar', 'issue_m_May', 'issue_m_Nov', 'issue_m_Oct', 'issue_m_Sep', 'loan_amnt', 'median_income_2016', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_sats', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_sats', 'num_tl_op_past_12m', 'open_acc', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'poverty_level_below_pct', 'pub_rec', 'pub_rec_bankruptcies', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_home_improvement', 'revol_bal', 'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1', 'sub_grade_D2', 'term', 'tot_cur_bal', 'total_acc', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'total_rev_hi_lim']
    



```
loan_variables_selected = []
for col in df_loan_accepted_census_cleaned.columns:
    if len([s for s in features_list_new if col.startswith(s)])>0:
        loan_variables_selected.append(col)

print('After hot-decoding, they corresponds to the following {} features from the original dataset.'.format(len(loan_variables_selected)))
print('')
print(loan_variables_selected)
```


    After hot-decoding, they corresponds to the following 54 features from the original dataset.
    
    ['loan_amnt', 'term', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mths_since_last_major_derog', 'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_sats', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_sats', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'Population', 'median_income_2016', 'female_pct', 'Black_pct', 'White_pct', 'Native_pct', 'Asian_pct', 'Hispanic_pct', 'household_family_pct', 'poverty_level_below_pct', 'Graduate_Degree_pct', 'employment_2016_rate']
    

**Insights on features importance**

Looking at the results above, we can bring in following conclusions:

- The term is the most important element each investor has to care about. 68-months loans are highly risky.
- The purpose of loan for credit cards payment brings more confidence to an investor.
- Borrowers who have 10 or more years verified working experience are the most trustworthy investment.
- Home ownership plays a significant role.
- The state of California is a significant factor to be considered when looking at the likelihood of Charged-Off
- Lenders should look at financial KPIs such as inq_last_6mths, num_tl_op_past_12m and acc_open_past_24mths; not just at FICO score, which are less relevant than these KPIs.
- Debt-to-income ratio and annual income can be missleading, and shoudn't be always considered as the most important factors.
- The time of the year when the loan is requested is not so relevant.

**Design Matrix with important features**

We can now create the new design matrix using the identified important features.



```
X_train_new = fs_model.transform(X_train)
X_val_new = fs_model.transform(X_val)
X_test_new = fs_model.transform(X_test)
```


**Principal Components Analysis**

We are interested in reducing the dimension of our data further by analysing its principal components. This will allow us to compress the important features into a reduced number of components.

We first start with scaling the data.



```
scaler = StandardScaler().fit(X_train_new)
X_train_scaled = scaler.transform(X_train_new)
X_val_scaled = scaler.transform(X_val_new)
X_test_scaled = scaler.transform(X_test_new)
```


We decompose the scaled data with PCA



```
n = X_train_new.shape[1]
pca_fit = PCA(n).fit(X_train_scaled)
```


In the plot below, we see that the dimension can be reduced to the number of components which explain at least 80% of variance in the data.



```
pca_var = np.cumsum(pca_fit.explained_variance_ratio_)
plt.scatter(range(1,n+1),pca_var)
plt.xlabel("PCA Dimensions")
plt.ylabel("Total Variance Captured")
plt.title("Variance Explained by PCA");
```



![png](4%29%20Models_files/4%29%20Models_89_0.png)


We can now rebuild our design matrix using the PCA components.



```
pca_fit = PCA(40).fit(X_train_scaled)
X_train_scaled_pca = pca_fit.transform(X_train_scaled)
X_val_scaled_pca = pca_fit.transform(X_val_scaled)
X_test_scaled_pca = pca_fit.transform(X_test_scaled)
```


## Classification Models

In the previous section we used a model-based approach for identifying the important features which most probably determine the failure or success of a loan.

Using the features selected, we will now investigate the performance of several models on the validation set via cross-validation. Each model will be trained on the training set. We will also check if dimension reduction via PCA improves the accuracy. 

At the end we will investigate if ensemble technique via stacking of base learners improves classification results.

### Training Data

For performance reason, we can resample the data and use a smaller volume for training our models. 

There are three versions: the original set, scaled set, scaled set with PCA.




```
print(" We have {} samples with {} features in our datasets.").format(X_train_new.shape[0], X_train_new.shape[1]))
```





    (90966, 1091)





```
from sklearn.utils import resample

frac = 1

X_train_small, y_train_small = resample(X_train_new, y_train, n_samples=round(y_train.shape[0]*frac))
X_val_small, y_val_small = resample(X_val_new, y_val, n_samples=round(y_train.shape[0]*frac))
X_test_small, y_test_small = resample(X_test_new, y_test, n_samples=round(y_train.shape[0]*frac))


"""np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_train_upsampled_smote_10.csv', 
           X_train_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_val_upsampled_smote_10.csv', 
           X_val_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_test_upsampled_smote_10.csv', 
           X_test_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_train_upsampled_smote_10.csv', 
           y_train_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_val_upsampled_smote_10.csv', 
           y_val_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_test_upsampled_smote_10.csv', 
           y_test_small, fmt='%.2f', delimiter=',', header='loan_status')
"""

X_train_scaled_small, y_train_scaled_small = resample(X_train_scaled, y_train, n_samples=round(y_train.shape[0]*frac))
X_val_scaled_small, y_val_scaled_small = resample(X_val_scaled, y_val, n_samples=round(y_train.shape[0]*frac))
X_test_scaled_small, y_test_scaled_small = resample(X_test_scaled, y_test, n_samples=round(y_train.shape[0]*frac))

"""
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_train_scaled_upsampled_smote_10.csv', 
           X_train_scaled_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_val_scaled_upsampled_smote_10.csv', 
           X_val_scaled_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_test_scaled_upsampled_smote_10.csv', 
           X_test_scaled_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_train_scaled_upsampled_smote_10.csv', 
           y_train_scaled_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_val_scaled_upsampled_smote_10.csv', 
           y_val_scaled_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_test_scaled_upsampled_smote_10.csv', 
           y_test_scaled_small, fmt='%.2f', delimiter=',', header='loan_status')
"""

X_train_scaled_pca_small, y_train_scaled_pca_small = resample(X_train_scaled_pca, y_train, n_samples=round(y_train.shape[0]*frac))
X_val_scaled_pca_small, y_val_scaled_pca_small = resample(X_val_scaled_pca, y_val, n_samples=round(y_train.shape[0]*frac))
X_test_scaled_pca_small, y_test_scaled_pca_small = resample(X_test_scaled_pca, y_test, n_samples=round(y_train.shape[0]*frac))

"""
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_train_scaled_pca_upsampled_smote_10.csv', 
           X_train_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_val_scaled_pca_upsampled_smote_10.csv', 
           X_val_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_test_scaled_pca_upsampled_smote_10.csv', 
           X_test_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_train_scaled_pca_upsampled_smote_10.csv', 
           y_train_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_val_scaled_pca_upsampled_smote_10.csv', 
           y_val_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')
np.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_test_scaled_pca_upsampled_smote_10.csv', 
           y_test_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')
"""
```





    "\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_train_scaled_pca_upsampled_smote_10.csv', \n           X_train_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_val_scaled_pca_upsampled_smote_10.csv', \n           X_val_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_X_test_scaled_pca_upsampled_smote_10.csv', \n           X_test_scaled_pca_small, fmt='%.2f', delimiter=',', header=','.join(X.columns.values))\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_train_scaled_pca_upsampled_smote_10.csv', \n           y_train_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_val_scaled_pca_upsampled_smote_10.csv', \n           y_val_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')\nnp.savetxt('df_loan_accepted_census_cleaned_closed_2007-2015_y_test_scaled_pca_upsampled_smote_10.csv', \n           y_test_scaled_pca_small, fmt='%.2f', delimiter=',', header='loan_status')\n"




### Scoring

We will work with the following metrics:

- **Recall or Sensitivity or TPR (True Positive Rate)**: Number of loans correctly identified as positive (fully paid) out of total true positives - TP/(TP+FN)
    
- **Specificity or TNR (True Negative Rate)**: Number of loans correctly identified as negative (charged-off) out of total negatives - TN/(TN+FP)

- **Precision**: Number of loans correctly identified as positive (fully paid) out of total items identified as positive - TP/(TP+FP)
    
- **False Positive Rate or Type I Error**: Number of loans wrongly identified as positive (fully paid) out of total true negatives - FP/(FP+TN)
    
- **False Negative Rate or Type II Error**: Number of loans wrongly identified as negative (charged-off) out of total true positives - FN/(FN+TP)

- A **Confusion Matrix**: visual representation of the number of TP, TN, FP and FN.

- **Accuracy**: Percentage of total items classified correctly - (TP+TN)/(N+P)

- **F1 Score**: Harmonic mean of precision and recall given by - F1 = 2xPrecisionxRecall /(Precision + Recall)

- **ROC-AUC Score**: Area under curve of sensitivity (TPR) vs. specificity (TNR).

- **Log-loss**: Probabilistic confidence of accuracy. High value of log-loss means that the absolute probabilities have big difference from actual labels.  

**Scoring in investment strategy**

https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428

If we choose an investment strategy that uses absolute probabilistic difference, then we will  look at log-loss with care. If the final class prediction is the most important outcome and we don’t want to tune probability threshold, we will rather use AUC score. But if the threshold is well tuned, then F1 will be the scoring to use.

In loan classification, where negative labels (charged-offs) are few, we would like our model to predict negative classes correctly and hence we will sometime prefer those models which are able to classify these negative labels. Log-loss usually fails to identify model which produces too many false negatives because the log-loss function is symmetric and does not differentiate between classes.  Both F1 score and ROC-AUC score can perform well for class imbalance. F1 is better suit for situations where the positive class is small. Since an investor would care more about the minority class (charged-off loans) in number independent of the fact whether it is positive or negative, then we think that ROC-AUC score would make sense as benchmark measure.

**Helper functions for scoring metrics**



```
# dataframe where we track all cross-validation scoring metrics
df_cv_scores = pd.DataFrame({'model':['dummy'], 'accuracy':[0], 'neg_log_loss':[0], 'f1':[0], 'roc_auc':[0]}, 
                            columns=['accuracy','neg_log_loss','f1','roc_auc'], index=['model'])
```




```
# function for computing 5-fold cross-validation scoring scores
def predict_evaluate_cv(model, X, y):
    score_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    score_log_loss = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss').mean()
    score_f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    score_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    df_cv_scores.loc[model.__class__.__name__] = [score_accuracy, -score_log_loss, score_f1, score_auc]
    print('K-fold cross-validation results:')
    print(model.__class__.__name__+" average accuracy is %2.3f" % score_accuracy)
    print(model.__class__.__name__+" average log_loss is %2.3f" % -score_log_loss)
    print(model.__class__.__name__+" average F1 is %2.3f" % score_f1)
    print(model.__class__.__name__+" average auc is %2.3f" % score_auc)
```




```
# function for computing the confusion matrix
def predict_evaluate_cm(model, X, y):
    classes = ['Fully Paid', 'Charged-Off']
    y_true, y_pred = y, model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #print(classification_report(y,y_pred))
```




```
# function for compution the roc plot
def predict_evaluate_roc(model, X, y):
    # source https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    [fpr, tpr, thr] = roc_curve(y, y_pred_proba)

    idx = np.min(np.where(tpr > 0.95))
    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()

    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
          "and a specificity of %.3f" % (1-fpr[idx]) + 
          ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
```




```
# global function for fitting, cross-validating and evaluating a given classifier
def fit_predict_evaluate(model, Xtrain, ytrain, Xval, yval):
    model.fit(Xtrain, ytrain)
    print(model.__class__.__name__+":")
    print('Accuracy, Training Set: {0:0.2%}'.format(model.score(Xtrain, ytrain)))
    predict_evaluate_cv(model, Xval, yval)
    predict_evaluate_cm(model, Xval, yval)
    predict_evaluate_roc(model, Xval, yval)
```


## Logistic Regression

We will start with a simple logistic regression model for predicting loan charge-off. The penalty parameter is found via cross-validation on the training set.



```
log_reg = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=5
        ,random_state=777
        ,max_iter=10000
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10
    )
```




```
fit_predict_evaluate(log_reg, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
```


    LogisticRegressionCV:
    Accuracy, Training Set: 67.51%
    K-fold cross-validation results:
    LogisticRegressionCV average accuracy is 0.816
    LogisticRegressionCV average log_loss is 0.433
    LogisticRegressionCV average F1 is 0.897
    LogisticRegressionCV average auc is 0.723
    


![png](4%29%20Models_files/4%29%20Models_105_1.png)



![png](4%29%20Models_files/4%29%20Models_105_2.png)


    Using a threshold of 0.227 guarantees a sensitivity of 0.950 and a specificity of 0.188, i.e. a false positive rate of 81.16%.
    

AUC is 0.80, there is 80% of chance that the logistic regression model will be able to distinguish between fully paid loans and charged-off loans.

**Logistic Regression with PCA**



```
fit_predict_evaluate(log_reg, X_train_scaled_pca_small, y_train_scaled_pca_small, X_val_scaled_pca_small, y_val_scaled_pca_small)
```


    LogisticRegressionCV:
    Accuracy, Training Set: 71.32%
    

    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1920: ChangedBehaviorWarning: The long-standing behavior to use the accuracy score has changed. The scoring parameter is now used. This warning will disappear in version 0.22.
      ChangedBehaviorWarning)
    

    K-fold cross-validation results:
    LogisticRegressionCV average accuracy is 0.812
    LogisticRegressionCV average log_loss is 0.444
    LogisticRegressionCV average F1 is 0.896
    LogisticRegressionCV average auc is 0.701
    


![png](4%29%20Models_files/4%29%20Models_108_3.png)


                  precision    recall  f1-score   support
    
               0       0.30      0.66      0.41     17137
               1       0.89      0.64      0.75     74046
    
       micro avg       0.65      0.65      0.65     91183
       macro avg       0.59      0.65      0.58     91183
    weighted avg       0.78      0.65      0.68     91183
    
    


![png](4%29%20Models_files/4%29%20Models_108_5.png)


    Using a threshold of 0.252 guarantees a sensitivity of 0.950 and a specificity of 0.159, i.e. a false positive rate of 84.07%.
    

PCA causes a decrease in average AUC.

## Random Forest

We will now rebuilt our random forest classifier, this time using the important features.



```
randomf_optim = RandomForestClassifier(n_estimators=200, max_depth=30)
```




```
fit_predict_evaluate(randomf_optim, X_train_small, y_train_small, X_val_small, y_val_small)
```


    RandomForestClassifier:
    Accuracy, Training Set: 100.00%
    K-fold cross-validation results:
    RandomForestClassifier average accuracy is 0.999
    RandomForestClassifier average log_loss is 0.015
    RandomForestClassifier average F1 is 0.999
    RandomForestClassifier average auc is 1.000
    


![png](4%29%20Models_files/4%29%20Models_112_1.png)



![png](4%29%20Models_files/4%29%20Models_112_2.png)


    Using a threshold of 0.565 guarantees a sensitivity of 0.951 and a specificity of 0.178, i.e. a false positive rate of 82.18%.
    

The random forest classifier gives the best accuracy so far.

## Boosting

In this section, we try the boosting technique for loan default prediction.



```
ab_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=60, learning_rate=0.05)
```




```
fit_predict_evaluate(ab_model, X_train_small, y_train_small, X_val_small, y_val_small)
```


    AdaBoostClassifier:
    Accuracy, Training Set: 90.78%
    



```
fig, ax = plt.subplots(1,2,figsize=(20,5))
train_scores = list(ab_model.staged_score(X_train,y_train))
test_scores = list(ab_model.staged_score(X_test, y_test))
ax[0].plot(train_scores,label='depth-{}'.format(depth))
ax[1].plot(test_scores,label='depth-{}'.format(depth))
ax[0].set_xlabel('number of iterations', fontsize=12)
ax[1].set_xlabel('number of iterations', fontsize=12)
ax[0].set_ylabel('Accuracy', fontsize=12)
ax[0].set_title("Variation of Accuracy with Iterations (training set)", fontsize=14)
ax[1].set_title("Variation of Accuracy with Iterations (validation set)", fontsize=14)
ax[0].legend(fontsize=12);
ax[1].legend(fontsize=12);    
```


### XG Boosting



```
xgb_model = XGBClassifier(learningrate =0.01, nestimators=100,
                    maxdepth=4, minchildweight=4, subsample=0.8, colsamplebytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scaleposweight=2,
                    seed=27)
```




```
fit_predict_evaluate(xgb_model, X_train_small, y_train_small, X_val_small, y_val_small)
```


    XGBClassifier:
    Accuracy, Training Set: 88.29%
    K-fold cross-validation results:
    XGBClassifier average accuracy is 0.813
    XGBClassifier average log_loss is 0.438
    XGBClassifier average F1 is 0.896
    XGBClassifier average auc is 0.717
    


![png](4%29%20Models_files/4%29%20Models_120_1.png)



![png](4%29%20Models_files/4%29%20Models_120_2.png)


    Using a threshold of 0.559 guarantees a sensitivity of 0.950 and a specificity of 0.180, i.e. a false positive rate of 81.98%.
    

## SVM



```
svm_model = SVC(gamma=0.1, C=0.01, kernel="linear")
```




```
fit_predict_evaluate(svm_model, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
```


    SVC:
    Accuracy, Training Set: 66.88%
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-285-f0d34d722ab7> in <module>()
    ----> 1 fit_predict_evaluate(svm_model, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
    

    <ipython-input-129-650159ccd3b5> in fit_predict_evaluate(model, Xtrain, ytrain, Xval, yval)
          3     print(model.__class__.__name__+":")
          4     print('Accuracy, Training Set: {0:0.2%}'.format(model.score(Xtrain, ytrain)))
    ----> 5     predict_evaluate_cv(model, Xval, yval)
          6     predict_evaluate_cm(model, Xval, yval)
          7     predict_evaluate_roc(model, Xval, yval)
    

    <ipython-input-133-dd2446e28dbe> in predict_evaluate_cv(model, X, y)
          1 def predict_evaluate_cv(model, X, y):
          2     score_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    ----> 3     score_log_loss = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss').mean()
          4     score_f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
          5     score_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, error_score)
        400                                 fit_params=fit_params,
        401                                 pre_dispatch=pre_dispatch,
    --> 402                                 error_score=error_score)
        403     return cv_results['test_score']
        404 
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in cross_validate(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, return_train_score, return_estimator, error_score)
        238             return_times=True, return_estimator=return_estimator,
        239             error_score=error_score)
    --> 240         for train, test in cv.split(X, y, groups))
        241 
        242     zipped_scores = list(zip(*scores))
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in __call__(self, iterable)
        915             # remaining jobs.
        916             self._iterating = False
    --> 917             if self.dispatch_one_batch(iterator):
        918                 self._iterating = self._original_iterator is not None
        919 
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in dispatch_one_batch(self, iterator)
        757                 return False
        758             else:
    --> 759                 self._dispatch(tasks)
        760                 return True
        761 
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in _dispatch(self, batch)
        714         with self._lock:
        715             job_idx = len(self._jobs)
    --> 716             job = self._backend.apply_async(batch, callback=cb)
        717             # A job can complete so quickly than its callback is
        718             # called before we get here, causing self._jobs to
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in apply_async(self, func, callback)
        180     def apply_async(self, func, callback=None):
        181         """Schedule a func to be run"""
    --> 182         result = ImmediateResult(func)
        183         if callback:
        184             callback(result)
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/_parallel_backends.py in __init__(self, batch)
        547         # Don't delay the application, to avoid keeping the input
        548         # arguments in memory
    --> 549         self.results = batch()
        550 
        551     def get(self):
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in __call__(self)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py in <listcomp>(.0)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, error_score)
        566         fit_time = time.time() - start_time
        567         # _score will return dict if is_multimetric is True
    --> 568         test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        569         score_time = time.time() - start_time - fit_time
        570         if return_train_score:
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in _score(estimator, X_test, y_test, scorer, is_multimetric)
        603     """
        604     if is_multimetric:
    --> 605         return _multimetric_score(estimator, X_test, y_test, scorer)
        606     else:
        607         if y_test is None:
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/model_selection/_validation.py in _multimetric_score(estimator, X_test, y_test, scorers)
        633             score = scorer(estimator, X_test)
        634         else:
    --> 635             score = scorer(estimator, X_test, y_test)
        636 
        637         if hasattr(score, 'item'):
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/metrics/scorer.py in __call__(self, clf, X, y, sample_weight)
        125         """
        126         y_type = type_of_target(y)
    --> 127         y_pred = clf.predict_proba(X)
        128         if y_type == "binary":
        129             if y_pred.shape[1] == 2:
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/svm/base.py in predict_proba(self)
        607         datasets.
        608         """
    --> 609         self._check_proba()
        610         return self._predict_proba
        611 
    

    ~/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/svm/base.py in _check_proba(self)
        574     def _check_proba(self):
        575         if not self.probability:
    --> 576             raise AttributeError("predict_proba is not available when "
        577                                  " probability=False")
        578         if self._impl not in ('c_svc', 'nu_svc'):
    

    AttributeError: predict_proba is not available when  probability=False


## QDA

In this section we try a QDA model.



```
qda_model = QuadraticDiscriminantAnalysis()
```




```
fit_predict_evaluate(qda_model, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
```


    QuadraticDiscriminantAnalysis:
    Accuracy, Training Set: 65.52%
    K-fold cross-validation results:
    QuadraticDiscriminantAnalysis average accuracy is 0.658
    QuadraticDiscriminantAnalysis average log_loss is 1.383
    QuadraticDiscriminantAnalysis average F1 is 0.760
    QuadraticDiscriminantAnalysis average auc is 0.679
    


![png](4%29%20Models_files/4%29%20Models_126_1.png)



![png](4%29%20Models_files/4%29%20Models_126_2.png)


    Using a threshold of 0.000 guarantees a sensitivity of 0.950 and a specificity of 0.131, i.e. a false positive rate of 86.87%.
    

## KNN



```
knn_model = KNeighborsClassifier(n_neighbors=3)
```




```
fit_predict_evaluate(knn_model, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
```


## Loan Description

https://www.thecut.com/2017/05/what-the-words-you-use-in-a-loan-application-reveal.html

Some studies suggest that words used on loan applications can predict the likelihood of charge-off.
In this section we use natural language processing algorithms to extract features from the loan title and description filled in by the borrower when requesting the loan. We then use a Naive Bayes classifier and a random forest for this task.



```
def clean_text(text):
    #https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
    # lower case
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove punctuation
    text = text.str.replace('[^\w\s]','') 
    # remove stop words    
    stop = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # correct spelling
    #from textblob import TextBlob
    #text = text.apply(lambda x: str(TextBlob(x).correct()))
    # lemmatization     
    text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return text
```




```
#>>> import nltk
#>>> nltk.download('wordnet')
#>>> nltk.download('stopwords')

df_desc = df_loan_accepted_census_cleaned.copy()
df_desc = df_desc[['title', 'desc', 'loan_status']]
df_desc.fillna('N/A', inplace=True)
df_desc['desc'] = df_desc['desc'] + ' - ' + df_desc['title']
df_desc = df_desc[df_desc.loan_status.isin(['Charged Off', 'Fully Paid'])]
df_desc.replace({'loan_status':{'Charged Off': 0, 'Fully Paid': 1}}, inplace=True)
df_desc.loan_status = df_desc.loan_status.astype('int')
df_desc.desc = clean_text(df_desc.desc).str.replace('na','')
```




```
X_desc = df_desc.title
y_desc = df_desc.loan_status
X_train_desc, X_test_desc, y_train_desc, y_test_desc = train_test_split(X_desc, y_desc, test_size= 0.2, random_state=13)
```




```
word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)

# fit and transform on it the training features
word_vectorizer.fit(X_train_desc)
X_train_desc_features = word_vectorizer.transform(X_train_desc)

#transform the test features to sparse matrix
X_test_desc_features = word_vectorizer.transform(X_test_desc)
```




```
X_test_desc_features.shape
```





    (155992, 13704)





```
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
```




```
fit_predict_evaluate(clf, X_train_desc_features, y_train_desc, X_test_desc_features, y_test_desc)
```


    MultinomialNB:
    Accuracy, Training Set: 81.38%
    K-fold cross-validation results:
    MultinomialNB average accuracy is 0.811
    MultinomialNB average log_loss is 0.493
    MultinomialNB average F1 is 0.896
    MultinomialNB average auc is 0.534
    


![png](4%29%20Models_files/4%29%20Models_138_1.png)



![png](4%29%20Models_files/4%29%20Models_138_2.png)


    Using a threshold of 0.798 guarantees a sensitivity of 0.958 and a specificity of 0.047, i.e. a false positive rate of 95.32%.
    



```
rf_desc = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
fit_predict_evaluate(rf_desc, X_train_desc_features, y_train_desc, X_test_desc_features, y_test_desc)
```


    RandomForestClassifier:
    Accuracy, Training Set: 81.17%
    K-fold cross-validation results:
    RandomForestClassifier average accuracy is 0.812
    RandomForestClassifier average log_loss is 0.482
    RandomForestClassifier average F1 is 0.896
    RandomForestClassifier average auc is 0.543
    


![png](4%29%20Models_files/4%29%20Models_139_1.png)



![png](4%29%20Models_files/4%29%20Models_139_2.png)


    Using a threshold of 0.805 guarantees a sensitivity of 0.991 and a specificity of 0.014, i.e. a false positive rate of 98.56%.
    

Both Naive Bayes and random forest did not find enough information that would clearly distinguish the two classes of loans.

## Stacking

We should consider stacking all the models obtained for achieving better prediction accuracy.

We begin with this simple approach of averaging base models. 



```
voting_clf = VotingClassifier(estimators=[
    ('rf', randomf_optim), 
    ('lr', log_reg),
    ('qda', qda_model)], voting='soft', flatten_transform=True)
voting_clf.fit(X_train_scaled, y_train)
```





    VotingClassifier(estimators=[('rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=30, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weig...rs=None, reg_param=0.0,
                   store_covariance=False, store_covariances=None, tol=0.0001))],
             flatten_transform=True, n_jobs=None, voting='soft', weights=None)





```
fit_predict_evaluate(voting_clf, X_train_scaled_small, y_train_scaled_small, X_val_scaled_small, y_val_scaled_small)
```


It seems even the simplest stacking approach improved the score of our best base learner. This encourages us to go further and explore a less simple stacking approach.

In this approach, we add a meta-model on averaged base models and use the out-of-folds predictions of these base models to train our meta-model. 



```
#https://towardsdatascience.com/predicting-loan-repayment-5df4e0023e92
models = {'rf': randomf_optim, 
        'lr': log_reg,
        'qda': qda_model}
```


We previously split the total training set into 5 disjoint folds (4 training and 1 holdout). We train the base models on the training folds and we predict on the holdout fold.



```
# Build first stack of base learners
first_stack = make_pipeline(voting_clf, FunctionTransformer(lambda X: X[:, 1::2]))
# Use CV to generate meta-features
meta_features = cross_val_predict(first_stack, X_train_scaled_small, y_train_scaled_small, cv=5, method="transform")
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    



```
# Refit the first stack on the full training set
first_stack.fit(X_train_scaled_small, y_train_scaled_small)
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    



```
# Fit the meta learner
logreg_clf = LogisticRegression(penalty="l2", C=100, fit_intercept=True)
second_stack = logreg_clf.fit(meta_features, y_train_scaled_small)
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    



```
# Plot ROC and PR curves using all models and test data
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, model in models.items():
    model_probs = model.predict_proba(X_val_scaled_small)[:, 1:]
    model_auc_score = roc_auc_score(y_val_scaled_small, model_probs)
    fpr, tpr, _ = roc_curve(y_val_scaled_small, model_probs)
    precision, recall, _ = precision_recall_curve(y_val_scaled_small, model_probs)
    axes[0].plot(fpr, tpr, label=f"{name}, auc = {model_auc_score:.3f}")
    axes[1].plot(recall, precision, label=f"{name}")

stacked_probs = second_stack.predict_proba(first_stack.transform(X_val_scaled_small))[:, 1:]
stacked_auc_score = roc_auc_score(y_val_scaled_small, stacked_probs)
fpr, tpr, _ = roc_curve(y_val_scaled_small, stacked_probs)
precision, recall, _ = precision_recall_curve(y_val_scaled_small, stacked_probs)
axes[0].plot(fpr, tpr, label=f"stacked_ensemble, auc = {stacked_auc_score:.3f}")
axes[1].plot(recall, precision, label="stacked_ensembe")
axes[0].legend(loc="lower right")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curve")
axes[1].legend()
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
axes[1].set_title("PR curve")
plt.tight_layout()
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/sklearn/preprocessing/_function_transformer.py:98: FutureWarning: The default validate=True will be replaced by validate=False in 0.22.
      "validate=False in 0.22.", FutureWarning)
    


![png](4%29%20Models_files/4%29%20Models_151_1.png)


Stacking did not improve the accuracy of our three base learners.

## Models Benchmark

All models investigated in this project are compared in the table below. The performance was achieved on the validation set using 5-fold cross-validation.



```
df_cv_scores
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>neg_log_loss</th>
      <th>f1</th>
      <th>roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>model</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LogisticRegressionCV</th>
      <td>0.812312</td>
      <td>-0.442843</td>
      <td>0.894996</td>
      <td>0.708746</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.657611</td>
      <td>1.382761</td>
      <td>0.759832</td>
      <td>0.678648</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.813156</td>
      <td>0.438469</td>
      <td>0.895662</td>
      <td>0.717397</td>
    </tr>
    <tr>
      <th>MultinomialNB</th>
      <td>0.810913</td>
      <td>0.492669</td>
      <td>0.895557</td>
      <td>0.533533</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.972509</td>
      <td>0.126344</td>
      <td>0.986063</td>
      <td>0.616578</td>
    </tr>
  </tbody>
</table>
</div>



**Performance on test set**

# Investment Strategy

Given the results from the previous section, we can know formulate investment strategies.

1. only A1, 36 month loans - that would be for CONSERVATIVE INVESTORS such as retirees
2. only A1-A5, 36 month loans - a little less conservative
3. only E1-E5, high risk and potentially high-reward (results will confirm or not) for SPECULATION

The function below fits a predictive model with a subset of data depending on the invesment strategy (sub grades and term), and it displays cross-validation score.



```
def simulate_strategy(df_loan_strategy, sub_grades=['A1'], term=36):
    df_loan_strategy = df_loan_strategy[df_loan_strategy.loan_status.isin(['Charged Off', 'Fully Paid'])]
    df_loan_strategy.replace({'loan_status':{'Charged Off': 0, 'Fully Paid': 1}}, inplace=True)
    df_loan_strategy.loan_status = df_loan_strategy.loan_status.astype('int')
    df_loan_strategy = df_loan_strategy[df_loan_strategy.term==term]
    df_loan_strategy = df_loan_strategy[df_loan_strategy.sub_grade.isin(sub_grades)]
    not_predictor_strategy = not_predictor + ['sub_grade','issue_m', 'addr_state', 'zip_code','earliest_cr_line']
    df_loan_strategy.drop(columns=list(set(not_predictor_strategy) & set(df_loan_strategy.columns.values)), inplace=True)
    df_loan_strategy = pd.get_dummies(df_loan_strategy, columns=['emp_length', 'home_ownership'], drop_first=True)
    df_loan_strategy = df_loan_strategy[list(set(features_list_new).intersection(set(df_loan_strategy.columns.values)))+['loan_status']]
    #df_loan_strategy = df_loan_strategy.sample(frac=.1)
    X, Y = df_loan_strategy[df_loan_strategy.columns.difference(['loan_status'])], df_loan_strategy['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=9001)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=9001)
    smote = SMOTE(ratio='minority')
    X_train, y_train = smote.fit_sample(X_train, y_train)
    #ros = RandomOverSampler()
    #X_train, y_train = ros.fit_sample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=10) # WILLIAM - change this model if necessary
    fit_predict_evaluate(model, X_train, y_train, X_val, y_val)
    return X_train, y_train, X_val, y_val, X_test, y_test
```




```
df_loan_strategy = df_loan_accepted_census_cleaned.copy()
X_train_strategy, y_train_strategy, X_val_strategy, y_val_strategy, X_test_strategy, y_test_strategy = simulate_strategy(df_loan_strategy, sub_grades=['A1'], term=36)
```


    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/generic.py:4550: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      regex=regex)
    /home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/generic.py:3643: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value
    

    RandomForestClassifier:
    Accuracy, Training Set: 97.74%
    K-fold cross-validation results:
    RandomForestClassifier average accuracy is 0.973
    RandomForestClassifier average log_loss is 0.126
    RandomForestClassifier average F1 is 0.986
    RandomForestClassifier average auc is 0.617
    


![png](4%29%20Models_files/4%29%20Models_160_2.png)



![png](4%29%20Models_files/4%29%20Models_160_3.png)


    Using a threshold of 0.607 guarantees a sensitivity of 0.951 and a specificity of 0.096, i.e. a false positive rate of 90.38%.
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-233-b45d28757120> in <module>()
          1 df_loan_strategy = df_loan_accepted_census_cleaned.copy()
    ----> 2 model_strategy, X_train_strategy, y_train_strategy, X_val_strategy, y_val_strategy, X_test_strategy, y_test_strategy = simulate_strategy(df_loan_strategy, sub_grades=['A1'], term=36)
    

    ValueError: not enough values to unpack (expected 7, got 6)

