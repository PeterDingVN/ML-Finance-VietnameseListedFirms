from .libs import *
from sklearn.impute import KNNImputer

# EDA function describe + skew, kurtosis
def eda_describe(df, columns=None):

  if not columns:
    # select only numeric col
    num_col1 = df.select_dtypes(include='number').columns

    # describe function
    df_describe = df[num_col1].describe().round(4)

    # check for kurtosis and skewness
    sk = stats.skew(df[num_col1], nan_policy='omit')
    k = stats.kurtosis(df[num_col1], nan_policy='omit')

    sk_k = pd.DataFrame([sk, k], columns = num_col1)
    sk_k = sk_k.apply(lambda x: round(x,4))

    result = pd.concat([df_describe, sk_k])
    result.rename(index = {0: 'skewness', 1: 'kurtosis'}, inplace = True)

  else:
    # select only numeric col
    num_col1 = df[columns].select_dtypes(include='number').columns

    # describe function
    df_describe = df[num_col1].describe().round(4)

    # check for kurtosis and skewness
    sk = stats.skew(df[num_col1], nan_policy='omit')
    k = stats.kurtosis(df[num_col1], nan_policy='omit')

    sk_k  = pd.DataFrame([sk,k], columns = num_col1)
    sk_k = sk_k.apply(lambda x: round(x,4))

    result = pd.concat([df_describe, sk_k])
    result.rename(index = {0: 'skewness', 1: 'kurtosis'}, inplace = True)


  return result.T

def select_data(df):
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

def impute(df):
  # requirement
  df_filled = df.copy()
  num_cols = df.select_dtypes(include='number').columns

  # set up
  imputer = KNNImputer(n_neighbors=3, weights='distance')

  # Fill
  df_filled[num_cols] = imputer.fit_transform(df_filled[num_cols])

  return df_filled