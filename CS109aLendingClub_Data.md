{:.no_toc}
*  
{: toc}



Lending Club publishes all its historical data since its inception in 2007. It provides two online, open access datasets for accepted and rejected loans from 2007 to 2018 Q2 for a period of 11.5 years in comma-separated values (CSV) format. Each accepted loan data set has 146 features for each of observation. With basic data cleaning, we removed the index column, columns with constant value and columns associated with 90% missing values that reduced the number of features to 106. Further we identified features which are strongly correlated (r=±0.8) that left us with 82 features on the accepted loan dataset.

To help us in our goal of analyzing fairness and interpretability we downloaded the census data about American people and US economy from United States Census Bureau. We have social, economic, and geographic data from 2016 grouped by zip code provided via Piazza by CS109a instructors. This dataset has 33120 zip codes described by 135 features. Trivial columns such as name, population and others were removed that left us with 85 features. Then the census data was merged with loan accepted data on zipcode ending with 167 features. Then we further used scikit learn feature selection on SelectPercentile reducing 141 predictors for our model analysis. For reconciliation, we downloaded data for all years and then we took a random sample of 10% and saved it to our own server “https://digintu.tech/tmp/cs109a/” stored in the file “loan_accepted_10.csv” for easy access. Also removed empty/duplicate rows that left us with 200K observations.

Second dataset provided online is the rejected loan information, again from 2007 to 2018 Q2 for 11.5 years. This dataset has 9 features. Each application of rejected loan has 9 features and the total amount was worth $22million not funded. These Rejected Loans data files contain the list and details of all loan applications that did not meet Lending Club's credit underwriting policy and the application was rejected.

We group the remaining features from loan accepted merged with census data into 5 classes:
● loan data: information about the loan at the moment when it was requested
● loan follow up: information about the loan's follow up throughout its term
● borrower demographics: information about the borrower
● borrower financial profile: financial background of the borrower at the moment when he requested the loan
● borrower financial profile follow up: changes in financial profile of the borrower throughout the loan term


**Helper functions**



```
"""
The following function returns the description of the features in lending club data.
Parameters
  df: dataframe containing the data
"""
def df_features_desc(df):
    df_cols_desc = load_data(['LCDataDictionary.csv'])
    desc = pd.DataFrame(df.columns).merge(df_cols_desc, how='left', left_on=0, right_on='LoanStatNew')[['LoanStatNew','Description']]
    desc = pd.DataFrame(df.dtypes).reset_index().merge(desc, how='right', left_on='index', right_on='LoanStatNew')[['LoanStatNew', 0,'Description']]
    desc = desc.rename(columns={0: 'Data Type', 'LoanStatNew':'Feature'})#.sort_values(by=['Feature'])
    desc = desc.dropna()
    pd.set_option('display.max_colwidth', -1)
    desc.style.set_properties({'text-align': 'left'})
    display(HTML(desc.to_html()))
```


## Accepted Loans

We load all loans accepted (approved) and funded on the LendingClub marketplace from 2007 to 2018 Q3 for a period of 11.5 years from separated files into the dataframe `df_loan_accepted`.



```
df_loan_accepted = load_data(['loan_accepted_10.csv'])
```




```
df_loan_accepted.shape
```





    (200409, 146)



There are approximately **2 millions accepted loans**, each of them has 146 columns. Many columns are however empty.



```
df_loan_accepted.describe()
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
      <th>Unnamed: 0</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>url</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>...</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200409.000000</td>
      <td>0.0</td>
      <td>200407.000000</td>
      <td>200407.000000</td>
      <td>200407.000000</td>
      <td>200407.000000</td>
      <td>2.004060e+05</td>
      <td>0.0</td>
      <td>200293.000000</td>
      <td>200402.000000</td>
      <td>...</td>
      <td>881.0</td>
      <td>881.000000</td>
      <td>881.0</td>
      <td>881.000000</td>
      <td>700.000000</td>
      <td>881.000000</td>
      <td>881.000000</td>
      <td>2619.000000</td>
      <td>2619.000000</td>
      <td>2619.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>99055.357324</td>
      <td>NaN</td>
      <td>14925.943330</td>
      <td>14920.633261</td>
      <td>14899.501524</td>
      <td>443.203570</td>
      <td>7.742113e+04</td>
      <td>NaN</td>
      <td>18.653801</td>
      <td>0.315266</td>
      <td>...</td>
      <td>3.0</td>
      <td>150.793394</td>
      <td>3.0</td>
      <td>12.542565</td>
      <td>443.717143</td>
      <td>11555.922758</td>
      <td>200.722781</td>
      <td>5043.492249</td>
      <td>47.868813</td>
      <td>12.512791</td>
    </tr>
    <tr>
      <th>std</th>
      <td>91099.127758</td>
      <td>NaN</td>
      <td>9046.369332</td>
      <td>9044.458980</td>
      <td>9048.247653</td>
      <td>264.271981</td>
      <td>7.869853e+04</td>
      <td>NaN</td>
      <td>12.279450</td>
      <td>0.869024</td>
      <td>...</td>
      <td>0.0</td>
      <td>117.160448</td>
      <td>0.0</td>
      <td>10.002764</td>
      <td>343.897939</td>
      <td>7079.323715</td>
      <td>197.927608</td>
      <td>3650.713791</td>
      <td>7.220576</td>
      <td>8.330692</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>16.850000</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>0.640000</td>
      <td>3.0</td>
      <td>0.000000</td>
      <td>1.920000</td>
      <td>55.730000</td>
      <td>0.010000</td>
      <td>221.260000</td>
      <td>20.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>36044.000000</td>
      <td>NaN</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>251.680000</td>
      <td>4.600000e+04</td>
      <td>NaN</td>
      <td>11.920000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>62.200000</td>
      <td>3.0</td>
      <td>0.000000</td>
      <td>183.735000</td>
      <td>6245.200000</td>
      <td>40.040000</td>
      <td>2265.470000</td>
      <td>45.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>74274.000000</td>
      <td>NaN</td>
      <td>12800.000000</td>
      <td>12800.000000</td>
      <td>12700.000000</td>
      <td>377.040000</td>
      <td>6.500000e+04</td>
      <td>NaN</td>
      <td>17.820000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>120.210000</td>
      <td>3.0</td>
      <td>13.000000</td>
      <td>352.815000</td>
      <td>10184.380000</td>
      <td>144.010000</td>
      <td>4248.000000</td>
      <td>45.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>122155.000000</td>
      <td>NaN</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>587.340000</td>
      <td>9.200000e+04</td>
      <td>NaN</td>
      <td>24.420000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>205.280000</td>
      <td>3.0</td>
      <td>21.000000</td>
      <td>602.970000</td>
      <td>15778.330000</td>
      <td>309.570000</td>
      <td>6855.900000</td>
      <td>50.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>421092.000000</td>
      <td>NaN</td>
      <td>40000.000000</td>
      <td>40000.000000</td>
      <td>40000.000000</td>
      <td>1719.830000</td>
      <td>9.300000e+06</td>
      <td>NaN</td>
      <td>999.000000</td>
      <td>21.000000</td>
      <td>...</td>
      <td>3.0</td>
      <td>828.490000</td>
      <td>3.0</td>
      <td>30.000000</td>
      <td>2066.880000</td>
      <td>36734.040000</td>
      <td>1377.170000</td>
      <td>28000.000000</td>
      <td>93.990000</td>
      <td>65.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 107 columns</p>
</div>



Below is a short description of columns in the accepted loan data set.



```
df_features_desc(df_loan_accepted)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Data Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>object</td>
      <td>A unique LC assigned ID for the loan listing.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>member_id</td>
      <td>float64</td>
      <td>A unique LC assigned Id for the borrower member.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>loan_amnt</td>
      <td>float64</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>funded_amnt</td>
      <td>float64</td>
      <td>The total amount committed to that loan at that point in time.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>funded_amnt_inv</td>
      <td>float64</td>
      <td>The total amount committed by investors for that loan at that point in time.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>term</td>
      <td>object</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>int_rate</td>
      <td>object</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>installment</td>
      <td>float64</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>grade</td>
      <td>object</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sub_grade</td>
      <td>object</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>10</th>
      <td>emp_title</td>
      <td>object</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>11</th>
      <td>emp_length</td>
      <td>object</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>home_ownership</td>
      <td>object</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>13</th>
      <td>annual_inc</td>
      <td>float64</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>verification_status</td>
      <td>object</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>15</th>
      <td>issue_d</td>
      <td>object</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>16</th>
      <td>loan_status</td>
      <td>object</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>17</th>
      <td>pymnt_plan</td>
      <td>object</td>
      <td>Indicates if a payment plan has been put in place for the loan</td>
    </tr>
    <tr>
      <th>18</th>
      <td>url</td>
      <td>float64</td>
      <td>URL for the LC page with listing data.</td>
    </tr>
    <tr>
      <th>19</th>
      <td>desc</td>
      <td>object</td>
      <td>Loan description provided by the borrower</td>
    </tr>
    <tr>
      <th>20</th>
      <td>purpose</td>
      <td>object</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>21</th>
      <td>title</td>
      <td>object</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>22</th>
      <td>zip_code</td>
      <td>object</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>addr_state</td>
      <td>object</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dti</td>
      <td>float64</td>
      <td>A ratio calculated using the borrowers total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowers self-reported monthly income.</td>
    </tr>
    <tr>
      <th>25</th>
      <td>delinq_2yrs</td>
      <td>float64</td>
      <td>The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years</td>
    </tr>
    <tr>
      <th>26</th>
      <td>earliest_cr_line</td>
      <td>object</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>27</th>
      <td>inq_last_6mths</td>
      <td>float64</td>
      <td>The number of inquiries in past 6 months (excluding auto and mortgage inquiries)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>mths_since_last_delinq</td>
      <td>float64</td>
      <td>The number of months since the borrower's last delinquency.</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mths_since_last_record</td>
      <td>float64</td>
      <td>The number of months since the last public record.</td>
    </tr>
    <tr>
      <th>30</th>
      <td>open_acc</td>
      <td>float64</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>31</th>
      <td>pub_rec</td>
      <td>float64</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>32</th>
      <td>revol_bal</td>
      <td>float64</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>33</th>
      <td>revol_util</td>
      <td>object</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>34</th>
      <td>total_acc</td>
      <td>float64</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>35</th>
      <td>initial_list_status</td>
      <td>object</td>
      <td>The initial listing status of the loan. Possible values are  W, F</td>
    </tr>
    <tr>
      <th>36</th>
      <td>out_prncp</td>
      <td>float64</td>
      <td>Remaining outstanding principal for total amount funded</td>
    </tr>
    <tr>
      <th>37</th>
      <td>out_prncp_inv</td>
      <td>float64</td>
      <td>Remaining outstanding principal for portion of total amount funded by investors</td>
    </tr>
    <tr>
      <th>38</th>
      <td>total_pymnt</td>
      <td>float64</td>
      <td>Payments received to date for total amount funded</td>
    </tr>
    <tr>
      <th>39</th>
      <td>total_pymnt_inv</td>
      <td>float64</td>
      <td>Payments received to date for portion of total amount funded by investors</td>
    </tr>
    <tr>
      <th>40</th>
      <td>total_rec_prncp</td>
      <td>float64</td>
      <td>Principal received to date</td>
    </tr>
    <tr>
      <th>41</th>
      <td>total_rec_int</td>
      <td>float64</td>
      <td>Interest received to date</td>
    </tr>
    <tr>
      <th>42</th>
      <td>total_rec_late_fee</td>
      <td>float64</td>
      <td>Late fees received to date</td>
    </tr>
    <tr>
      <th>43</th>
      <td>recoveries</td>
      <td>float64</td>
      <td>post charge off gross recovery</td>
    </tr>
    <tr>
      <th>44</th>
      <td>collection_recovery_fee</td>
      <td>float64</td>
      <td>post charge off collection fee</td>
    </tr>
    <tr>
      <th>45</th>
      <td>last_pymnt_d</td>
      <td>object</td>
      <td>Last month payment was received</td>
    </tr>
    <tr>
      <th>46</th>
      <td>last_pymnt_amnt</td>
      <td>float64</td>
      <td>Last total payment amount received</td>
    </tr>
    <tr>
      <th>47</th>
      <td>next_pymnt_d</td>
      <td>object</td>
      <td>Next scheduled payment date</td>
    </tr>
    <tr>
      <th>48</th>
      <td>last_credit_pull_d</td>
      <td>object</td>
      <td>The most recent month LC pulled credit for this loan</td>
    </tr>
    <tr>
      <th>49</th>
      <td>collections_12_mths_ex_med</td>
      <td>float64</td>
      <td>Number of collections in 12 months excluding medical collections</td>
    </tr>
    <tr>
      <th>50</th>
      <td>mths_since_last_major_derog</td>
      <td>float64</td>
      <td>Months since most recent 90-day or worse rating</td>
    </tr>
    <tr>
      <th>51</th>
      <td>policy_code</td>
      <td>float64</td>
      <td>publicly available policy_code=1\Nnew products not publicly available policy_code=2publicly available policy_code=1\Nnew products not publicly available policy_code=2publicly available policy_code=1\nnew products not publicly available policy_code=2publicly available policy_code=1\nnew products not publicly available policy_code=2publicly available policy_code=1\nnew products not publicly available policy_code=2</td>
    </tr>
    <tr>
      <th>52</th>
      <td>application_type</td>
      <td>object</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>53</th>
      <td>annual_inc_joint</td>
      <td>float64</td>
      <td>The combined self-reported annual income provided by the co-borrowers during registration</td>
    </tr>
    <tr>
      <th>54</th>
      <td>dti_joint</td>
      <td>float64</td>
      <td>A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income</td>
    </tr>
    <tr>
      <th>55</th>
      <td>acc_now_delinq</td>
      <td>float64</td>
      <td>The number of accounts on which the borrower is now delinquent.</td>
    </tr>
    <tr>
      <th>56</th>
      <td>tot_coll_amt</td>
      <td>float64</td>
      <td>Total collection amounts ever owed</td>
    </tr>
    <tr>
      <th>57</th>
      <td>tot_cur_bal</td>
      <td>float64</td>
      <td>Total current balance of all accounts</td>
    </tr>
    <tr>
      <th>58</th>
      <td>open_acc_6m</td>
      <td>float64</td>
      <td>Number of open trades in last 6 months</td>
    </tr>
    <tr>
      <th>59</th>
      <td>open_act_il</td>
      <td>float64</td>
      <td>Number of currently active installment trades</td>
    </tr>
    <tr>
      <th>60</th>
      <td>open_il_12m</td>
      <td>float64</td>
      <td>Number of installment accounts opened in past 12 months</td>
    </tr>
    <tr>
      <th>61</th>
      <td>open_il_24m</td>
      <td>float64</td>
      <td>Number of installment accounts opened in past 24 months</td>
    </tr>
    <tr>
      <th>62</th>
      <td>mths_since_rcnt_il</td>
      <td>float64</td>
      <td>Months since most recent installment accounts opened</td>
    </tr>
    <tr>
      <th>63</th>
      <td>total_bal_il</td>
      <td>float64</td>
      <td>Total current balance of all installment accounts</td>
    </tr>
    <tr>
      <th>64</th>
      <td>il_util</td>
      <td>float64</td>
      <td>Ratio of total current balance to high credit/credit limit on all install acct</td>
    </tr>
    <tr>
      <th>65</th>
      <td>open_rv_12m</td>
      <td>float64</td>
      <td>Number of revolving trades opened in past 12 months</td>
    </tr>
    <tr>
      <th>66</th>
      <td>open_rv_24m</td>
      <td>float64</td>
      <td>Number of revolving trades opened in past 24 months</td>
    </tr>
    <tr>
      <th>67</th>
      <td>max_bal_bc</td>
      <td>float64</td>
      <td>Maximum current balance owed on all revolving accounts</td>
    </tr>
    <tr>
      <th>68</th>
      <td>all_util</td>
      <td>float64</td>
      <td>Balance to credit limit on all trades</td>
    </tr>
    <tr>
      <th>69</th>
      <td>inq_fi</td>
      <td>float64</td>
      <td>Number of personal finance inquiries</td>
    </tr>
    <tr>
      <th>70</th>
      <td>total_cu_tl</td>
      <td>float64</td>
      <td>Number of finance trades</td>
    </tr>
    <tr>
      <th>71</th>
      <td>inq_last_12m</td>
      <td>float64</td>
      <td>Number of credit inquiries in past 12 months</td>
    </tr>
    <tr>
      <th>72</th>
      <td>acc_open_past_24mths</td>
      <td>float64</td>
      <td>Number of trades opened in past 24 months.</td>
    </tr>
    <tr>
      <th>73</th>
      <td>avg_cur_bal</td>
      <td>float64</td>
      <td>Average current balance of all accounts</td>
    </tr>
    <tr>
      <th>74</th>
      <td>bc_open_to_buy</td>
      <td>float64</td>
      <td>Total open to buy on revolving bankcards.</td>
    </tr>
    <tr>
      <th>75</th>
      <td>bc_util</td>
      <td>float64</td>
      <td>Ratio of total current balance to high credit/credit limit for all bankcard accounts.</td>
    </tr>
    <tr>
      <th>76</th>
      <td>chargeoff_within_12_mths</td>
      <td>float64</td>
      <td>Number of charge-offs within 12 months</td>
    </tr>
    <tr>
      <th>77</th>
      <td>delinq_amnt</td>
      <td>float64</td>
      <td>The past-due amount owed for the accounts on which the borrower is now delinquent.</td>
    </tr>
    <tr>
      <th>78</th>
      <td>mo_sin_old_il_acct</td>
      <td>float64</td>
      <td>Months since oldest bank installment account opened</td>
    </tr>
    <tr>
      <th>79</th>
      <td>mo_sin_old_rev_tl_op</td>
      <td>float64</td>
      <td>Months since oldest revolving account opened</td>
    </tr>
    <tr>
      <th>80</th>
      <td>mo_sin_rcnt_rev_tl_op</td>
      <td>float64</td>
      <td>Months since most recent revolving account opened</td>
    </tr>
    <tr>
      <th>81</th>
      <td>mo_sin_rcnt_tl</td>
      <td>float64</td>
      <td>Months since most recent account opened</td>
    </tr>
    <tr>
      <th>82</th>
      <td>mort_acc</td>
      <td>float64</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>83</th>
      <td>mths_since_recent_bc</td>
      <td>float64</td>
      <td>Months since most recent bankcard account opened.</td>
    </tr>
    <tr>
      <th>84</th>
      <td>mths_since_recent_bc_dlq</td>
      <td>float64</td>
      <td>Months since most recent bankcard delinquency</td>
    </tr>
    <tr>
      <th>85</th>
      <td>mths_since_recent_inq</td>
      <td>float64</td>
      <td>Months since most recent inquiry.</td>
    </tr>
    <tr>
      <th>86</th>
      <td>mths_since_recent_revol_delinq</td>
      <td>float64</td>
      <td>Months since most recent revolving delinquency.</td>
    </tr>
    <tr>
      <th>87</th>
      <td>num_accts_ever_120_pd</td>
      <td>float64</td>
      <td>Number of accounts ever 120 or more days past due</td>
    </tr>
    <tr>
      <th>88</th>
      <td>num_actv_bc_tl</td>
      <td>float64</td>
      <td>Number of currently active bankcard accounts</td>
    </tr>
    <tr>
      <th>89</th>
      <td>num_actv_rev_tl</td>
      <td>float64</td>
      <td>Number of currently active revolving trades</td>
    </tr>
    <tr>
      <th>90</th>
      <td>num_bc_sats</td>
      <td>float64</td>
      <td>Number of satisfactory bankcard accounts</td>
    </tr>
    <tr>
      <th>91</th>
      <td>num_bc_tl</td>
      <td>float64</td>
      <td>Number of bankcard accounts</td>
    </tr>
    <tr>
      <th>92</th>
      <td>num_il_tl</td>
      <td>float64</td>
      <td>Number of installment accounts</td>
    </tr>
    <tr>
      <th>93</th>
      <td>num_op_rev_tl</td>
      <td>float64</td>
      <td>Number of open revolving accounts</td>
    </tr>
    <tr>
      <th>94</th>
      <td>num_rev_accts</td>
      <td>float64</td>
      <td>Number of revolving accounts</td>
    </tr>
    <tr>
      <th>95</th>
      <td>num_rev_tl_bal_gt_0</td>
      <td>float64</td>
      <td>Number of revolving trades with balance &gt;0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>num_sats</td>
      <td>float64</td>
      <td>Number of satisfactory accounts</td>
    </tr>
    <tr>
      <th>97</th>
      <td>num_tl_120dpd_2m</td>
      <td>float64</td>
      <td>Number of accounts currently 120 days past due (updated in past 2 months)</td>
    </tr>
    <tr>
      <th>98</th>
      <td>num_tl_30dpd</td>
      <td>float64</td>
      <td>Number of accounts currently 30 days past due (updated in past 2 months)</td>
    </tr>
    <tr>
      <th>99</th>
      <td>num_tl_90g_dpd_24m</td>
      <td>float64</td>
      <td>Number of accounts 90 or more days past due in last 24 months</td>
    </tr>
    <tr>
      <th>100</th>
      <td>num_tl_op_past_12m</td>
      <td>float64</td>
      <td>Number of accounts opened in past 12 months</td>
    </tr>
    <tr>
      <th>101</th>
      <td>pct_tl_nvr_dlq</td>
      <td>float64</td>
      <td>Percent of trades never delinquent</td>
    </tr>
    <tr>
      <th>102</th>
      <td>percent_bc_gt_75</td>
      <td>float64</td>
      <td>Percentage of all bankcard accounts &gt; 75% of limit.</td>
    </tr>
    <tr>
      <th>103</th>
      <td>pub_rec_bankruptcies</td>
      <td>float64</td>
      <td>Number of public record bankruptcies</td>
    </tr>
    <tr>
      <th>104</th>
      <td>tax_liens</td>
      <td>float64</td>
      <td>Number of tax liens</td>
    </tr>
    <tr>
      <th>105</th>
      <td>tot_hi_cred_lim</td>
      <td>float64</td>
      <td>Total high credit/credit limit</td>
    </tr>
    <tr>
      <th>106</th>
      <td>total_bal_ex_mort</td>
      <td>float64</td>
      <td>Total credit balance excluding mortgage</td>
    </tr>
    <tr>
      <th>107</th>
      <td>total_bc_limit</td>
      <td>float64</td>
      <td>Total bankcard high credit/credit limit</td>
    </tr>
    <tr>
      <th>108</th>
      <td>total_il_high_credit_limit</td>
      <td>float64</td>
      <td>Total installment high credit/credit limit</td>
    </tr>
    <tr>
      <th>109</th>
      <td>sec_app_open_act_il</td>
      <td>float64</td>
      <td>Number of currently active installment trades at time of application for the secondary applicant</td>
    </tr>
    <tr>
      <th>110</th>
      <td>hardship_flag</td>
      <td>object</td>
      <td>Flags whether or not the borrower is on a hardship plan</td>
    </tr>
    <tr>
      <th>111</th>
      <td>hardship_type</td>
      <td>object</td>
      <td>Describes the hardship plan offering</td>
    </tr>
    <tr>
      <th>112</th>
      <td>hardship_reason</td>
      <td>object</td>
      <td>Describes the reason the hardship plan was offered</td>
    </tr>
    <tr>
      <th>113</th>
      <td>hardship_status</td>
      <td>object</td>
      <td>Describes if the hardship plan is active, pending, canceled, completed, or broken</td>
    </tr>
    <tr>
      <th>114</th>
      <td>deferral_term</td>
      <td>float64</td>
      <td>Amount of months that the borrower is expected to pay less than the contractual monthly payment amount due to a hardship plan</td>
    </tr>
    <tr>
      <th>115</th>
      <td>hardship_amount</td>
      <td>float64</td>
      <td>The interest payment that the borrower has committed to make each month while they are on a hardship plan</td>
    </tr>
    <tr>
      <th>116</th>
      <td>hardship_start_date</td>
      <td>object</td>
      <td>The start date of the hardship plan period</td>
    </tr>
    <tr>
      <th>117</th>
      <td>hardship_end_date</td>
      <td>object</td>
      <td>The end date of the hardship plan period</td>
    </tr>
    <tr>
      <th>118</th>
      <td>payment_plan_start_date</td>
      <td>object</td>
      <td>The day the first hardship plan payment is due. For example, if a borrower has a hardship plan period of 3 months, the start date is the start of the three-month period in which the borrower is allowed to make interest-only payments.</td>
    </tr>
    <tr>
      <th>119</th>
      <td>hardship_length</td>
      <td>float64</td>
      <td>The number of months the borrower will make smaller payments than normally obligated due to a hardship plan</td>
    </tr>
    <tr>
      <th>120</th>
      <td>hardship_dpd</td>
      <td>float64</td>
      <td>Account days past due as of the hardship plan start date</td>
    </tr>
    <tr>
      <th>121</th>
      <td>hardship_loan_status</td>
      <td>object</td>
      <td>Loan Status as of the hardship plan start date</td>
    </tr>
    <tr>
      <th>122</th>
      <td>orig_projected_additional_accrued_interest</td>
      <td>float64</td>
      <td>The original projected additional interest amount that will accrue for the given hardship payment plan as of the Hardship Start Date. This field will be null if the borrower has broken their hardship payment plan.</td>
    </tr>
    <tr>
      <th>123</th>
      <td>hardship_payoff_balance_amount</td>
      <td>float64</td>
      <td>The payoff balance amount as of the hardship plan start date</td>
    </tr>
    <tr>
      <th>124</th>
      <td>hardship_last_payment_amount</td>
      <td>float64</td>
      <td>The last payment amount as of the hardship plan start date</td>
    </tr>
    <tr>
      <th>125</th>
      <td>disbursement_method</td>
      <td>object</td>
      <td>The method by which the borrower receives their loan. Possible values are: CASH, DIRECT_PAY</td>
    </tr>
    <tr>
      <th>126</th>
      <td>debt_settlement_flag</td>
      <td>object</td>
      <td>Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company.</td>
    </tr>
    <tr>
      <th>127</th>
      <td>debt_settlement_flag_date</td>
      <td>object</td>
      <td>The most recent date that the Debt_Settlement_Flag has been set</td>
    </tr>
    <tr>
      <th>128</th>
      <td>settlement_status</td>
      <td>object</td>
      <td>The status of the borrowers settlement plan. Possible values are: COMPLETE, ACTIVE, BROKEN, CANCELLED, DENIED, DRAFT</td>
    </tr>
    <tr>
      <th>129</th>
      <td>settlement_date</td>
      <td>object</td>
      <td>The date that the borrower agrees to the settlement plan</td>
    </tr>
    <tr>
      <th>130</th>
      <td>settlement_amount</td>
      <td>float64</td>
      <td>The loan amount that the borrower has agreed to settle for</td>
    </tr>
    <tr>
      <th>131</th>
      <td>settlement_percentage</td>
      <td>float64</td>
      <td>The settlement amount as a percentage of the payoff balance amount on the loan</td>
    </tr>
    <tr>
      <th>132</th>
      <td>settlement_term</td>
      <td>float64</td>
      <td>The number of months that the borrower will be on the settlement plan</td>
    </tr>
  </tbody>
</table>


## Rejected Loans

We load all loans rejected (not funded) on the LendingClub marketplace from 2007 to 2018 Q3 for a period of 11.5 years into the dataframe `df_loan_rejected`.



```
df_loan_rejected = pd.read_csv('data/RejectStatsA.csv', skiprows=(1))
```




```
df_loan_rejected.shape
```





    (755491, 9)



There are approximately 22 millions of rejected loans, each of them has 9 features.



```
df_loan_rejected.describe()
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
      <th>Amount Requested</th>
      <th>Risk_Score</th>
      <th>Policy Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.554910e+05</td>
      <td>731562.000000</td>
      <td>755491.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.291072e+04</td>
      <td>590.995754</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.068035e+04</td>
      <td>179.254816</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000e+03</td>
      <td>571.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000e+04</td>
      <td>644.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000e+04</td>
      <td>685.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.400000e+06</td>
      <td>850.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Census Data
The United States Census Bureau provides data about the American people and economy. We have social, economic, and geographic data from 2016 grouped by zip code provided via Piazza by CS109a instructors.

We load census data from 2016 into the dataframe `df_census`. 




```
df_census = pd.read_csv('data/zipcode_demographics_2016_USA.csv')
```




```
df_census.shape
```





    (33120, 135)



There are 33120 zip codes, each of them described by 135 statistics.



```
df_census.describe()
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
      <th>Unnamed: 0</th>
      <th>Population</th>
      <th>zip code tabulation area</th>
      <th>owner_renter_total</th>
      <th>owner_occupied</th>
      <th>renter_occupied</th>
      <th>abroad_year_ago_total</th>
      <th>abroad_year_ago_puerto_rico</th>
      <th>abroad_year_ago_us_islands</th>
      <th>abroad_year_ago_foreign</th>
      <th>...</th>
      <th>60_to_75k_2016_pct</th>
      <th>75_to_100k_2016_pct</th>
      <th>100_to_150k_2016_pct</th>
      <th>150_to_200k_2016_pct</th>
      <th>over_200k_2016_pct</th>
      <th>No_Diploma_pct</th>
      <th>High_school_pct</th>
      <th>Some_college_pct</th>
      <th>Bachelors_Degree_pct</th>
      <th>Graduate_Degree_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33120.000000</td>
      <td>33120.000000</td>
      <td>33120.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>32989.000000</td>
      <td>...</td>
      <td>23055.000000</td>
      <td>23055.000000</td>
      <td>23055.000000</td>
      <td>23055.000000</td>
      <td>23055.000000</td>
      <td>32765.000000</td>
      <td>32765.000000</td>
      <td>32765.000000</td>
      <td>32765.000000</td>
      <td>32765.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16559.500000</td>
      <td>9724.409300</td>
      <td>49666.334209</td>
      <td>9298.151808</td>
      <td>6073.192549</td>
      <td>3224.959259</td>
      <td>61.003031</td>
      <td>2.476250</td>
      <td>0.529843</td>
      <td>57.996938</td>
      <td>...</td>
      <td>0.098562</td>
      <td>0.107134</td>
      <td>0.097093</td>
      <td>0.034172</td>
      <td>0.032929</td>
      <td>0.010476</td>
      <td>0.290389</td>
      <td>0.232441</td>
      <td>0.148160</td>
      <td>0.084170</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9561.064794</td>
      <td>14358.657599</td>
      <td>27564.925769</td>
      <td>13855.431955</td>
      <td>8730.614992</td>
      <td>6313.579477</td>
      <td>170.145836</td>
      <td>21.042147</td>
      <td>5.755175</td>
      <td>164.081808</td>
      <td>...</td>
      <td>0.180024</td>
      <td>0.188002</td>
      <td>0.176695</td>
      <td>0.105539</td>
      <td>0.109215</td>
      <td>0.019820</td>
      <td>0.119239</td>
      <td>0.091032</td>
      <td>0.099925</td>
      <td>0.086035</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>601.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8279.750000</td>
      <td>718.000000</td>
      <td>26634.750000</td>
      <td>667.000000</td>
      <td>501.000000</td>
      <td>121.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.218567</td>
      <td>0.186317</td>
      <td>0.081940</td>
      <td>0.032538</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16559.500000</td>
      <td>2807.500000</td>
      <td>49739.000000</td>
      <td>2645.000000</td>
      <td>1984.000000</td>
      <td>538.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.033608</td>
      <td>0.039894</td>
      <td>0.018367</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.005961</td>
      <td>0.288293</td>
      <td>0.230056</td>
      <td>0.127752</td>
      <td>0.058981</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24839.250000</td>
      <td>13177.750000</td>
      <td>72123.500000</td>
      <td>12484.000000</td>
      <td>8334.000000</td>
      <td>3206.000000</td>
      <td>35.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>...</td>
      <td>0.119956</td>
      <td>0.133992</td>
      <td>0.125463</td>
      <td>0.022970</td>
      <td>0.011533</td>
      <td>0.013532</td>
      <td>0.357277</td>
      <td>0.273011</td>
      <td>0.196078</td>
      <td>0.106354</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33119.000000</td>
      <td>115104.000000</td>
      <td>99929.000000</td>
      <td>113403.000000</td>
      <td>81331.000000</td>
      <td>87101.000000</td>
      <td>3661.000000</td>
      <td>1205.000000</td>
      <td>292.000000</td>
      <td>3661.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 134 columns</p>
</div>




In this section we describe basic operations for cleaning our dataset.




```
df_loan_accepted_cleaned = df_loan_accepted.copy()
```


### Remove index column



```
df_loan_accepted_cleaned.drop(columns=['Unnamed: 0'], inplace=True)
```


### Remove columns with a constant value



```
cols_unique = list(df_loan_accepted_cleaned.columns[df_loan_accepted_cleaned.nunique()==1])
print('The following features with constant value were removed', cols_unique)
df_loan_accepted_cleaned.drop(cols_unique, axis=1, inplace=True)
print('{} features left, out of {} in the original dataset'.format(df_loan_accepted_cleaned.shape[1], df_loan_accepted.shape[1]))
```


    The following features with constant value were removed ['policy_code', 'hardship_type', 'deferral_term', 'hardship_length']
    141 features left, out of 146 in the original dataset
    

### Remove columns associated with over 90% missing values



```
df_missing = (df_loan_accepted_cleaned.isnull().sum()/df_loan_accepted_cleaned.shape[0]).to_frame('perc_missing').reset_index()
cols_missing = list(df_missing[df_missing.perc_missing>0.9]['index'])
print('The following features with over 90% missing values were removed.')
print(df_missing[df_missing.perc_missing>0.9])
df_loan_accepted_cleaned.drop(cols_missing, axis=1, inplace=True)
print('')
print('{} features left, out of {} in the original dataset'.format(df_loan_accepted_cleaned.shape[1], df_loan_accepted.shape[1]))
```


    The following features with over 90% missing values were removed.
                                              index  perc_missing
    0    id                                          0.999990    
    1    member_id                                   1.000000    
    18   url                                         1.000000    
    19   desc                                        0.936799    
    52   annual_inc_joint                            0.957427    
    53   dti_joint                                   0.957427    
    54   verification_status_joint                   0.957971    
    110  revol_bal_joint                             0.963744    
    111  sec_app_earliest_cr_line                    0.963744    
    112  sec_app_inq_last_6mths                      0.963744    
    113  sec_app_mort_acc                            0.963744    
    114  sec_app_open_acc                            0.963744    
    115  sec_app_revol_util                          0.964383    
    116  sec_app_open_act_il                         0.963744    
    117  sec_app_num_rev_accts                       0.963744    
    118  sec_app_chargeoff_within_12_mths            0.963744    
    119  sec_app_collections_12_mths_ex_med          0.963744    
    120  sec_app_mths_since_last_major_derog         0.987675    
    122  hardship_reason                             0.995604    
    123  hardship_status                             0.995604    
    124  hardship_amount                             0.995604    
    125  hardship_start_date                         0.995604    
    126  hardship_end_date                           0.995604    
    127  payment_plan_start_date                     0.995604    
    128  hardship_dpd                                0.995604    
    129  hardship_loan_status                        0.995604    
    130  orig_projected_additional_accrued_interest  0.996507    
    131  hardship_payoff_balance_amount              0.995604    
    132  hardship_last_payment_amount                0.995604    
    135  debt_settlement_flag_date                   0.986932    
    136  settlement_status                           0.986932    
    137  settlement_date                             0.986932    
    138  settlement_amount                           0.986932    
    139  settlement_percentage                       0.986932    
    140  settlement_term                             0.986932    
    
    106 features left, out of 146 in the original dataset
    

### Remove duplicate rows



```
orig_rows_count = df_loan_accepted_cleaned.shape[0]
df_loan_accepted_cleaned.drop_duplicates(inplace=True)
print("{} duplicated rows were removed.".format(orig_rows_count-df_loan_accepted_cleaned.shape[0]))
```


    1 duplicated rows were removed.
    

### Remove empty rows



```
orig_rows_count = df_loan_accepted_cleaned.shape[0]
df_loan_accepted_cleaned.dropna(inplace=True, how='all')
print("{} empty rows were removed.".format(orig_rows_count-df_loan_accepted_cleaned.shape[0]))
```


    1 empty rows were removed.
    


## Variables Groups

We divide the columns in the following groups:

- **loan data**: information about the loan at the moment when it was requested
- **loan followup**: information about the loan's followup throughout its term
- **borrower demographics**: information about the borrower
- **borrower financial profile**: financial background of the borrower at the moment when he requested the loan
- **borrower financial profile followup**: changes in financial profile of the borrower throughout the loan term



```
cols_loan_data = ['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','grade','sub_grade','issue_d','loan_status','purpose','title','initial_list_status','application_type','disbursement_method']
cols_loan_followup = ['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','debt_settlement_flag','pymnt_plan']
cols_borrower_demographics = ['emp_title','emp_length','home_ownership','annual_inc','verification_status', 'zip_code','addr_state']
cols_borrower_finance_profile = ['dti','delinq_2yrs','earliest_cr_line','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_coll_amt','mort_acc','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','hardship_flag']
cols_borrower_finance_profile_followup = ['last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog','acc_now_delinq','tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m']
```


## Loan Data

In this section we manually clean-up come features describing the resquest of loan by borrowers.

**Issue date**

Let's convert the loan's issue date into a datatime type and add month and quarter.



```
df_loan_accepted_cleaned['issue_q'] = pd.to_datetime(df_loan_accepted_cleaned.issue_d, format='%b-%Y').dt.to_period('Q')
df_loan_accepted_cleaned['issue_m'] = df_loan_accepted_cleaned.issue_d.str.replace(r'-\d+', '')
```


**Loan status**

Delinquency happens when a borrower fails to pay the minimum amount for an outstanding debt. In the countplot below we can see the amount of loans that incurred in any stage of delinquency, according to the definitions used by Lending Club.

    Charged Off — defaulted loans for which there is no expectation from the lender in recovering the debt
    Default — borrower has failed to pay his obligations for more than 120 days
    Late — borrower has failed to pay his obligations for 31 to 120 days
    Grace Period — borrower still has time to pay his obligations without being considered delinquent
    Late — payment is late by 16 to 30 days
    
The count of loans within each stage is given below.



```
df_loan_accepted_cleaned.loan_status.value_counts()
```





    Fully Paid                                             89859
    Current                                                83297
    Charged Off                                            22881
    Late (31-120 days)                                     2255 
    In Grace Period                                        1344 
    Late (16-30 days)                                      498  
    Does not meet the credit policy. Status:Fully Paid     198  
    Does not meet the credit policy. Status:Charged Off    71   
    Default                                                4    
    Name: loan_status, dtype: int64



Very few old loans have the status 'Does not meet the credit policy' and will not be considered in our project.



```
df_loan_accepted_cleaned.drop(df_loan_accepted_cleaned[df_loan_accepted_cleaned.loan_status=='Does not meet the credit policy. Status:Fully Paid'].index, inplace=True)
df_loan_accepted_cleaned.drop(df_loan_accepted_cleaned[df_loan_accepted_cleaned.loan_status=='Does not meet the credit policy. Status:Charged Off'].index, inplace=True)
```


We add a new feature for successfully paid loans.



```
df_loan_accepted_cleaned['success'] = df_loan_accepted_cleaned['loan_status']
df_loan_accepted_cleaned.replace({'success':{'Charged Off': 0,
                                             'Fully Paid': 1, 
                                             'Current': 2, 
                                             'In Grace Period': 3,
                                             'Late (16-30 days)': 4,
                                             'Late (31-120 days)': 5,
                                             'Default': 6 }}, inplace=True)
df_loan_accepted_cleaned['success'] = df_loan_accepted_cleaned['success'].astype('int')
```


**Term and interest rate**

We turn the term and interest rate into numbers.



```
df_loan_accepted_cleaned.term.unique()
```





    array([' 36 months', ' 60 months'], dtype=object)





```
df_loan_accepted_cleaned.term.replace(' 36 months', 36, inplace=True)
df_loan_accepted_cleaned.term.replace(' 60 months', 60, inplace=True)
df_loan_accepted_cleaned.term = df_loan_accepted_cleaned.term.astype('int')
```




```
df_loan_accepted_cleaned.term.unique()
```





    array([36, 60], dtype=int64)





```
df_loan_accepted_cleaned.int_rate.head()
```





    0     13.35%
    1     14.08%
    2     11.99%
    3      8.18%
    4     16.99%
    Name: int_rate, dtype: object





```
df_loan_accepted_cleaned.int_rate = df_loan_accepted_cleaned.int_rate.str[:-1]
df_loan_accepted_cleaned.int_rate = df_loan_accepted_cleaned.int_rate.astype('float32')
```




```
df_loan_accepted_cleaned.int_rate.head()
```





    0    13.35
    1    14.08
    2    11.99
    3    8.18 
    4    16.99
    Name: int_rate, dtype: float32



**Loan amounts**

We will transform the amounts into integers



```
df_loan_accepted_cleaned.loan_amnt = df_loan_accepted_cleaned.loan_amnt.astype('int')
```


We will drop rows where `loan_amnt`, `funded_amnt`, `funded_amnt_inv` or `installment` is missing. We transform the amounts into integers.



```
df_loan_accepted_cleaned = df_loan_accepted_cleaned[df_loan_accepted_cleaned.loan_amnt.notnull() &
                                                    df_loan_accepted_cleaned.funded_amnt.notnull() & 
                                                    df_loan_accepted_cleaned.funded_amnt_inv.notnull() & 
                                                    df_loan_accepted_cleaned.installment.notnull()]
```




```
df_loan_accepted_cleaned.funded_amnt_inv = df_loan_accepted_cleaned.loan_amnt.astype('int')
```


**Title**

We will replace missing `title` values with NA for not available.



```
df_loan_accepted_cleaned.replace({'title': {np.nan: 'N/A'}}, inplace=True)
```


There are thousands of distinct titles entered by borrowers for their loan.



```
df_loan_accepted_cleaned.title.unique().shape
```





    (8638,)



 We will prepare the titles for natural language features extration.



```
def clean_text(text):
    #https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
    # lower case
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove punctuation
    text = text.str.replace('[^\w\s]','') 
    # remove stop words
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # correct spelling
    #from textblob import TextBlob
    #text = text.apply(lambda x: str(TextBlob(x).correct()))
    # lemmatization 
    from textblob import Word
    text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return text
```




```
df_loan_accepted_cleaned.title = clean_text(df_loan_accepted_cleaned.title)
```




```
df_loan_accepted_cleaned.title.unique()
```





    array(['car financing', '', 'debt consolidation', ..., 'envelope system',
           'medical bill debt consolidation', 'beckys wedding'], dtype=object)



**Miscellenous**

The columns `initial_list_status`,`application_type`,`disbursement_method` describe two categories each, one of them being largely dominant.



```
df_loan_accepted_cleaned.initial_list_status.value_counts()
```





    w    130974
    f    69164 
    Name: initial_list_status, dtype: int64





```
df_loan_accepted_cleaned.application_type.value_counts()
```





    Individual    191606
    Joint App     8532  
    Name: application_type, dtype: int64





```
df_loan_accepted_cleaned.disbursement_method.value_counts()
```





    Cash         197328
    DirectPay    2810  
    Name: disbursement_method, dtype: int64



We remove those columns.



```
df_loan_accepted_cleaned.drop(columns=['initial_list_status','application_type','disbursement_method'], inplace=True)
```


## Loan Followup

In this section we manually clean-up the information about the current status of an active loan. These columns tell for example how much was paid back, when the next payment is to be expected etc.

For the goal of our project, we will consider `total_rec_late_fee` which indicate that the borrower had issues with paying the installment at some point in the past. This information could help computing the probability of charge-off or default for ongoing loans. 

We will also consider features such as `last_pymnt_d`, `total_rec_prncp`, `total_rec_int`, `last_pymnt_d` for computing the return of investment for closed loans.

We delete the remaining follow-up columns.



```
df_loan_accepted_cleaned.drop(columns=['out_prncp','out_prncp_inv','debt_settlement_flag','pymnt_plan','recoveries','hardship_flag'], inplace=True)
```


## Borrower demographics

In this section we manually clean-up data related to the borrower.

**Employment Title**

We replace missing employment title by 'N/A'. 



```
df_loan_accepted_cleaned.replace({'emp_title': {np.nan: 'N/A'}}, inplace=True)
```


**Employment Length**

Missing employment length is replaced by 0.



```
df_loan_accepted_cleaned.emp_length.fillna(value=0,inplace=True)
df_loan_accepted_cleaned.emp_length.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df_loan_accepted_cleaned.emp_length.replace(to_replace='self-employed', value='0', inplace=True, regex=True)
df_loan_accepted_cleaned.emp_length.replace(to_replace='', value='0', inplace=True, regex=True)
df_loan_accepted_cleaned.emp_length = df_loan_accepted_cleaned.emp_length.astype(int)
```


**Annual Income**

There are too many outliers in `annual_inc`, which should be removed.



```
orig_rows_count = df_loan_accepted_cleaned.shape[0]
df_loan_accepted_cleaned = df_loan_accepted_cleaned[~(df_loan_accepted_cleaned.annual_inc > 250000)]
print("{} rows removed with annual_inc > 250000.".format(orig_rows_count-df_loan_accepted_cleaned.shape[0]))
```


    2213 rows removed with annual_inc > 250000.
    

## Borrower Financial Profile

In this section we clean columns which describe the credit history of the borrower.

**Revolving Line Utilization Rate**

We remove the '%' sign from the revolving line utilization rate and turn the column to float datatype.



```
df_loan_accepted_cleaned.revol_util = df_loan_accepted_cleaned.revol_util.fillna('0%')
df_loan_accepted_cleaned.revol_util = df_loan_accepted_cleaned.revol_util.str[:,-1]
df_loan_accepted_cleaned.revol_util = df_loan_accepted_cleaned.revol_util.astype('float32')
```


**Credit History**

It is safe to impute missing values for the following columns with zero.



```
cols = ['dti','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_coll_amt','mort_acc','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit']

df_loan_accepted_cleaned[cols] = df_loan_accepted_cleaned[cols].fillna(0)
```


**Trade Delinquency**

The percent of trades never delinquent `pct_tl_nvr_dlq` is set to 100% if it is missing.



```
df_loan_accepted_cleaned.pct_tl_nvr_dlq.fillna(100, inplace=True)
```


## Borrower Financial Profile Followup

In this section we manually clean some columns which contain a more current financial information about the borrower.

`last_credit_pull_d` indicates how old the financial information about the borrower is. It is safe to drop this feature.



```
df_loan_accepted_cleaned.drop(columns=['last_credit_pull_d'], inplace=True)
```


We will set the ratio of total current balance to high credit/credit limit for all bankcard accounts to 100% when missing.



```
df_loan_accepted_cleaned.bc_util.fillna(100, inplace=True)
```


It is safe to impute missing values for the remaining columns with zero.



```
cols = ['collections_12_mths_ex_med','mths_since_last_major_derog','acc_now_delinq','tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m']

df_loan_accepted_cleaned[cols] = df_loan_accepted_cleaned[cols].fillna(0)
```


## Correlation

In this section, we check the correlation between all remaining features.



```
def find_high_correlated_features(frame):
    new_corr = frame.corr()
    new_corr.loc[:,:] = np.tril(new_corr, k=-1) 
    new_corr = new_corr.stack()
    print(new_corr[(new_corr > 0.8) | (new_corr < -0.8)])

```




```
find_high_correlated_features(df_loan_accepted_cleaned)   
```


    funded_amnt                 loan_amnt            0.999743
    funded_amnt_inv             loan_amnt            1.000000
                                funded_amnt          0.999743
    installment                 loan_amnt            0.945965
                                funded_amnt          0.946320
                                funded_amnt_inv      0.945965
    open_rv_24m                 open_rv_12m          0.834431
    all_util                    il_util              0.828947
    avg_cur_bal                 tot_cur_bal          0.823542
    num_actv_rev_tl             num_actv_bc_tl       0.830213
    num_bc_sats                 num_actv_bc_tl       0.842139
    num_op_rev_tl               open_acc             0.801644
                                num_actv_rev_tl      0.817740
    num_rev_accts               num_bc_tl            0.853224
                                num_op_rev_tl        0.808960
    num_rev_tl_bal_gt_0         num_actv_bc_tl       0.824096
                                num_actv_rev_tl      0.984166
                                num_op_rev_tl        0.821867
    num_sats                    open_acc             0.954830
                                num_op_rev_tl        0.839493
    num_tl_30dpd                acc_now_delinq       0.817772
    tot_hi_cred_lim             tot_cur_bal          0.982046
    total_bc_limit              total_rev_hi_lim     0.813747
                                bc_open_to_buy       0.848926
    total_il_high_credit_limit  total_bal_ex_mort    0.889408
    dtype: float64
    

As shown above, `installment` carries the same information as the `funded_amnt_in`. Similarly `open_il_24m` and `open_il_12m` are highly correlated. Same for `open_rv_24m` and `open_rv_12m`; `all_util` and `il_util`; `total_rev_hi_lim` and `revol_bal`; `bc_util` and `revol_util`; `avg_cur_bal` and `tot_cur_bal`; `num_actv_bc_tl` and `num_actv_bc_tl`; `num_tl_30dpd` and `acc_now_delinq`. We consider dropping some of those columns below.



```
df_loan_accepted_cleaned.drop(columns=['open_il_12m','open_rv_12m','il_util','revol_bal', 
                                       'revol_util','avg_cur_bal','num_actv_bc_tl','num_tl_30dpd'], inplace=True)
```


## Census Data

After cleaning the loan acceptance data, we will now clean and add census data to it.

### Cleaning census data



```
df_census_cleaned = df_census.copy()
```


We keep the following columns and remove the rest from the census data.



```
census_cols = ['Population', 'zip code tabulation area', 'median_income_2016', 
               'male_pct', 'female_pct', 
               'Black_pct', 'Native_pct', 'Asian_pct', 'Hispanic_pct', 
               'household_family_pct', 'poverty_level_below_pct', 'Graduate_Degree_pct', 'employment_2016_rate']
df_census_cleaned = df_census_cleaned[census_cols]
```


We remove rows with median income less than zero.



```
df_census_cleaned = df_census_cleaned[df_census_cleaned.median_income_2016>0]
```


We first fill missing values with zeros in the census dataset.



```
df_census_cleaned.isnull().sum()
```





    Population                  0   
    zip code tabulation area    0   
    median_income_2016          0   
    male_pct                    0   
    female_pct                  0   
    Black_pct                   0   
    Native_pct                  0   
    Asian_pct                   0   
    Hispanic_pct                0   
    household_family_pct        0   
    poverty_level_below_pct     9760
    Graduate_Degree_pct         0   
    employment_2016_rate        0   
    dtype: int64





```
df_census_cleaned.fillna(0, inplace=True)
```


Add a new column with zip codes in the format 123XX



```
df_census_cleaned['zip_code'] = df_census_cleaned['zip code tabulation area'].astype('str')
df_census_cleaned['zip_code'] = df_census_cleaned['zip_code'].str.pad(5, 'left', '0')
df_census_cleaned['zip_code'] = df_census_cleaned['zip_code'].str.slice(0,3)
df_census_cleaned['zip_code'] = df_census_cleaned['zip_code'].str.pad(5, 'right', 'x')
df_census_cleaned.drop(columns=['zip code tabulation area'], inplace=True)
```


Aggregate by zipcode and take the mean of census values



```
df_census_cleaned = df_census_cleaned.groupby(['zip_code']).mean().reset_index()
```




```
df_census_cleaned.tail()
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
      <th>zip_code</th>
      <th>Population</th>
      <th>median_income_2016</th>
      <th>male_pct</th>
      <th>female_pct</th>
      <th>Black_pct</th>
      <th>Native_pct</th>
      <th>Asian_pct</th>
      <th>Hispanic_pct</th>
      <th>household_family_pct</th>
      <th>poverty_level_below_pct</th>
      <th>Graduate_Degree_pct</th>
      <th>employment_2016_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>884</th>
      <td>995xx</td>
      <td>6268.461538</td>
      <td>59252.903846</td>
      <td>0.526823</td>
      <td>0.473177</td>
      <td>0.020715</td>
      <td>0.459778</td>
      <td>0.040214</td>
      <td>0.049570</td>
      <td>0.684839</td>
      <td>0.063769</td>
      <td>0.069403</td>
      <td>0.760864</td>
    </tr>
    <tr>
      <th>885</th>
      <td>996xx</td>
      <td>2560.727273</td>
      <td>50849.181818</td>
      <td>0.534283</td>
      <td>0.465717</td>
      <td>0.007251</td>
      <td>0.565686</td>
      <td>0.034245</td>
      <td>0.027964</td>
      <td>0.695934</td>
      <td>0.089168</td>
      <td>0.046758</td>
      <td>0.749790</td>
    </tr>
    <tr>
      <th>886</th>
      <td>997xx</td>
      <td>2019.910448</td>
      <td>48305.447761</td>
      <td>0.546723</td>
      <td>0.453277</td>
      <td>0.007017</td>
      <td>0.652971</td>
      <td>0.010724</td>
      <td>0.022896</td>
      <td>0.697308</td>
      <td>0.029398</td>
      <td>0.047139</td>
      <td>0.733550</td>
    </tr>
    <tr>
      <th>887</th>
      <td>998xx</td>
      <td>3639.714286</td>
      <td>61540.857143</td>
      <td>0.518490</td>
      <td>0.481510</td>
      <td>0.015683</td>
      <td>0.196095</td>
      <td>0.031797</td>
      <td>0.044007</td>
      <td>0.663403</td>
      <td>0.036125</td>
      <td>0.076276</td>
      <td>0.829295</td>
    </tr>
    <tr>
      <th>888</th>
      <td>999xx</td>
      <td>2420.888889</td>
      <td>49593.444444</td>
      <td>0.532552</td>
      <td>0.467448</td>
      <td>0.001796</td>
      <td>0.285796</td>
      <td>0.015533</td>
      <td>0.024939</td>
      <td>0.558452</td>
      <td>0.123077</td>
      <td>0.061948</td>
      <td>0.822142</td>
    </tr>
  </tbody>
</table>
</div>



### Loan and census data consolidation



```
 df_loan_accepted_census_cleaned = pd.merge(df_loan_accepted_cleaned, df_census_cleaned, on='zip_code')
```




```
print('The merged loan and census dataset has {} features'.format(df_loan_accepted_census_cleaned.shape[1]))
```


    The merged loan and census dataset has 99 features
    

Let's save the final accepted loan dataset augmented with census data to disk.

Merge accepted loan data with census data using the zip code



```
df_loan_accepted_census_cleaned.to_csv('df_loan_accepted_census_cleaned.csv')
```



In this section we will have a look at the rejected loan requests and do basic cleaning.




```
df_loan_rejected.head()
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
      <th>Amount Requested</th>
      <th>Application Date</th>
      <th>Loan Title</th>
      <th>Risk_Score</th>
      <th>Debt-To-Income Ratio</th>
      <th>Zip Code</th>
      <th>State</th>
      <th>Employment Length</th>
      <th>Policy Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000.0</td>
      <td>2007-05-26</td>
      <td>Wedding Covered but No Honeymoon</td>
      <td>693.0</td>
      <td>10%</td>
      <td>481xx</td>
      <td>NM</td>
      <td>4 years</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000.0</td>
      <td>2007-05-26</td>
      <td>Consolidating Debt</td>
      <td>703.0</td>
      <td>10%</td>
      <td>010xx</td>
      <td>MA</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11000.0</td>
      <td>2007-05-27</td>
      <td>Want to consolidate my debt</td>
      <td>715.0</td>
      <td>10%</td>
      <td>212xx</td>
      <td>MD</td>
      <td>1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6000.0</td>
      <td>2007-05-27</td>
      <td>waksman</td>
      <td>698.0</td>
      <td>38.64%</td>
      <td>017xx</td>
      <td>MA</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1500.0</td>
      <td>2007-05-27</td>
      <td>mdrigo</td>
      <td>509.0</td>
      <td>9.43%</td>
      <td>209xx</td>
      <td>MD</td>
      <td>&lt; 1 year</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```
df_loan_rejected_cleaned = df_loan_rejected.copy()
```


We rename the columns in order to be able to merge the data with accepted loans later.



```
df_loan_rejected_cleaned.rename(columns={'Amount Requested':'loan_amnt',
                                         'Application Date': 'issue_d',
                                         'Risk_Score': 'risk_score',
                                         'Debt-To-Income Ratio': 'dti',
                                         'Zip Code':'zip_code',
                                         'State': 'addr_state',
                                         'Employment Length': 'emp_length',
                                         'Loan Title': 'title'},  inplace=True)
```


We remove columns with constant value



```
cols_unique = list(df_loan_rejected_cleaned.columns[df_loan_rejected_cleaned.nunique()==1])
print('Following columns with constant value were removed.')
print(cols_unique)
df_loan_rejected_cleaned.drop(cols_unique, axis=1, inplace=True)
```


    Following columns with constant value were removed.
    ['Policy Code']
    

We remove duplicated rows



```
orig_rows_count = df_loan_rejected_cleaned.shape[0]
df_loan_rejected_cleaned.drop_duplicates(inplace=True)
print("{} duplicated rows were removed.".format(orig_rows_count-df_loan_rejected_cleaned.shape[0]))
```


    1379 duplicated rows were removed.
    

We remove rows with empty Risk Score, Zip Code or State



```
df_loan_rejected_cleaned = df_loan_rejected_cleaned[df_loan_rejected_cleaned.risk_score.notnull() & 
                                                    df_loan_rejected_cleaned.zip_code.notnull() & 
                                                    df_loan_rejected_cleaned.addr_state.notnull()]
```


We prepare the loan title for natural language feature extraction.



```
df_loan_rejected_cleaned.replace({'title': {np.nan: 'N/A'}}, inplace=True)
df_loan_rejected_cleaned.title = clean_text(df_loan_rejected_cleaned.title)
```


We transform employment length to integer.



```
df_loan_rejected_cleaned.emp_length.fillna(value=0,inplace=True)
df_loan_rejected_cleaned.emp_length.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df_loan_rejected_cleaned.emp_length.replace(to_replace='', value='0', inplace=True, regex=True)
df_loan_rejected_cleaned.emp_length = df_loan_rejected_cleaned.emp_length.astype(int)
```


Transform debt to income ratio to float.



```
df_loan_rejected_cleaned.dti = df_loan_rejected_cleaned.dti.str.replace('%','')
df_loan_rejected_cleaned.dti = df_loan_rejected_cleaned.dti.astype('float32')
```


Add application month and quarter.



```
df_loan_rejected_cleaned['issue_m']= pd.to_datetime(df_loan_rejected_cleaned.issue_d, format='%Y-%m-%d').dt.strftime("%b")
df_loan_rejected_cleaned['issue_q']= pd.to_datetime(df_loan_rejected_cleaned.issue_d, format='%Y-%m-%d').dt.to_period('Q')
```


Loan and census data consolidation



```
 df_loan_rejected_census_cleaned = pd.merge(df_loan_rejected_cleaned, df_census_cleaned, on='zip_code')
```




```
df_loan_rejected_census_cleaned.head()
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
      <th>loan_amnt</th>
      <th>issue_d</th>
      <th>title</th>
      <th>risk_score</th>
      <th>dti</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>emp_length</th>
      <th>issue_m</th>
      <th>issue_q</th>
      <th>...</th>
      <th>male_pct</th>
      <th>female_pct</th>
      <th>Black_pct</th>
      <th>Native_pct</th>
      <th>Asian_pct</th>
      <th>Hispanic_pct</th>
      <th>household_family_pct</th>
      <th>poverty_level_below_pct</th>
      <th>Graduate_Degree_pct</th>
      <th>employment_2016_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000.0</td>
      <td>2007-05-26</td>
      <td>wedding covered honeymoon</td>
      <td>693.0</td>
      <td>10.00</td>
      <td>481xx</td>
      <td>NM</td>
      <td>4</td>
      <td>May</td>
      <td>2007Q2</td>
      <td>...</td>
      <td>0.494315</td>
      <td>0.505685</td>
      <td>0.071903</td>
      <td>0.003197</td>
      <td>0.032006</td>
      <td>0.03931</td>
      <td>0.671069</td>
      <td>0.225222</td>
      <td>0.127648</td>
      <td>0.804903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17000.0</td>
      <td>2007-06-18</td>
      <td>caterik</td>
      <td>628.0</td>
      <td>22.76</td>
      <td>481xx</td>
      <td>MI</td>
      <td>1</td>
      <td>Jun</td>
      <td>2007Q2</td>
      <td>...</td>
      <td>0.494315</td>
      <td>0.505685</td>
      <td>0.071903</td>
      <td>0.003197</td>
      <td>0.032006</td>
      <td>0.03931</td>
      <td>0.671069</td>
      <td>0.225222</td>
      <td>0.127648</td>
      <td>0.804903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3000.0</td>
      <td>2007-06-20</td>
      <td>joelbacon</td>
      <td>683.0</td>
      <td>4.69</td>
      <td>481xx</td>
      <td>OH</td>
      <td>2</td>
      <td>Jun</td>
      <td>2007Q2</td>
      <td>...</td>
      <td>0.494315</td>
      <td>0.505685</td>
      <td>0.071903</td>
      <td>0.003197</td>
      <td>0.032006</td>
      <td>0.03931</td>
      <td>0.671069</td>
      <td>0.225222</td>
      <td>0.127648</td>
      <td>0.804903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2300.0</td>
      <td>2007-07-13</td>
      <td>chrissy</td>
      <td>448.0</td>
      <td>6.24</td>
      <td>481xx</td>
      <td>MI</td>
      <td>9</td>
      <td>Jul</td>
      <td>2007Q3</td>
      <td>...</td>
      <td>0.494315</td>
      <td>0.505685</td>
      <td>0.071903</td>
      <td>0.003197</td>
      <td>0.032006</td>
      <td>0.03931</td>
      <td>0.671069</td>
      <td>0.225222</td>
      <td>0.127648</td>
      <td>0.804903</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8000.0</td>
      <td>2007-09-04</td>
      <td>secure216</td>
      <td>468.0</td>
      <td>19.73</td>
      <td>481xx</td>
      <td>MS</td>
      <td>1</td>
      <td>Sep</td>
      <td>2007Q3</td>
      <td>...</td>
      <td>0.494315</td>
      <td>0.505685</td>
      <td>0.071903</td>
      <td>0.003197</td>
      <td>0.032006</td>
      <td>0.03931</td>
      <td>0.671069</td>
      <td>0.225222</td>
      <td>0.127648</td>
      <td>0.804903</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Let's save the final rejected loan dataset augmented with census data to disk.



```
df_loan_rejected_census_cleaned.to_csv('df_loan_rejected_census_cleaned.csv')
```

