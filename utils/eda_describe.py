from libs import stats, pd


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