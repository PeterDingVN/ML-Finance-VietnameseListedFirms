from .libs import *
from sklearn.impute import KNNImputer


def eda_describe(df: pd.DataFrame) -> pd.DataFrame:
  '''
  param df: input a dataframe df or df[[...]]
  return: summary table of stats

  Example:
    eda_describe(df[['col1', 'col2']])
    -> return:
            mean    25%     50%     75%       ....
    col 1    20     10      19      25
    col 2    25     ...
  '''

  num_col1 = df.select_dtypes(include='number').columns

  df_describe = df[num_col1].describe().round(4)

  sk = stats.skew(df[num_col1], nan_policy='omit')
  k = stats.kurtosis(df[num_col1], nan_policy='omit')

  sk_k = pd.DataFrame([sk, k], columns = num_col1)
  sk_k = sk_k.apply(lambda x: round(x,4))

  result = pd.concat([df_describe, sk_k])
  result.rename(index = {0: 'skewness', 1: 'kurtosis'}, inplace = True)
  return result.T



def select_data(df: pd.DataFrame) -> pd.DataFrame:
  '''
  param: dataframe -> df or df[[]]
  return: new dataframe with only companies satisfying the condition

  Example:
    A   B    C    D
    1   1    NaN  NaN
    1   4    NaN  NaN
    1   7    3    7
  NaN   6    5    10

  Return -> only row 3 and 4 are kept because NaN < 2 condition
    A   B    C    D
    1   7    3    7
  NaN   6    5    10

  Note: the func automatically set min NaN <= 6 and len(year) >=4, meaning companies
  with data on at least 4 consecutive years and each obs has no more than 6 NaN would be chosen

  '''

  data = []
  try:
    df = df[df.isna().sum(axis=1) <= 6]
    for i, grp in df.groupby('company'):
      year = grp['year'].to_list()
      year.sort()
      if len(year) >= 4 and set(np.diff(year)) == {1}:
        data.append(grp)
    data_fin = pd.concat(data)
    return data_fin

  except ValueError as e:
    return e



def impute(df: pd.DataFrame) -> pd.DataFrame:
  '''
  param df: dataframe -> df or df[[]]
  return: fully filled dataframe using KNN

  Example:
    A   B    C    D
    1   1    NaN  NaN
    1   4    NaN  NaN
    1   7    3    7
  NaN   6    5    10

  Output:
    A   B    C    D
    1   1    4    5
    1   4    7    2
    1   7    3    7
    1   6    5    10

  '''

  df_filled = df.copy()
  num_cols = df.select_dtypes(include='number').columns

  imputer = KNNImputer(n_neighbors=3, weights='distance')

  df_filled[num_cols] = imputer.fit_transform(df_filled[num_cols])

  return df_filled