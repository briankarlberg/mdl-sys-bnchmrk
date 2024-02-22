# To be called as an import to 

# Traspose long-form X-file into biomarkers as columns,
def extract(df_lite):
    import time
    start = time.time()
    data_type = df_lite.columns[2] # extract marker values

    dot_T = df_lite.pivot_table(
        index='improve_sample_id',
        columns='entrez_id',
        values=data_type,
        aggfunc='mean'             # average duplicate values
    )

    end = time.time()
    wall_clock = end - start
    return str(round(wall_clock / 60, 2)) + ' minutes', dot_T

# Extract ids and biomarker values
def df_check(X_n):
    df_lite = X_n.iloc[:, :3] # cut the last two columns, source and study
    size = f"{df_lite.shape[0]:,}"
    na_count = f"{df_lite.isna().sum().sum():,}"
    inf_count = f"{df_lite.isin([np.inf, -np.inf]).sum().sum():,}"
    return df_lite, size, na_count, inf_count

# dot_T = g(d_typ, dot_T.copy())
def g(d_typ, df):
    """
    Checks the data types of columns and index in a DataFrame and prints informative messages.

    Args:
        df (pandas.DataFrame): The DataFrame to check.

    Returns:
        None
    """

    if df.columns.dtype == 'float64' and df.index.dtype == 'float64':
        print('both float')
        df = float_to_string(d_typ, df)
    elif df.columns.dtype == 'float64' and df.index.dtype == 'int':
        print('columns are float, index are int')
        df = indx_int_colm_flt(d_typ, df)
    elif df.columns.dtype == 'int' and df.index.dtype == 'float':
        print('columns are int, index are float, fail, write another function')
        # forth function
    elif df.columns.dtype == 'int' and df.index.dtype == 'int':
        print('columns are int, index are int')
        df = int_to_string(d_typ, df)
    else:
        print('non int or float dtype detected')
    return df

def int_to_string(d_typ, dot_T):
    dot_T.columns = dot_T.columns.map(str)
    dot_T.columns = ['entrz_' + d_typ + i for i in dot_T.columns] #
    dot_T.columns.name = 'entrez_id'

    dot_T.index = dot_T.index.map(str)
    dot_T.index = ['smpl_id_' + i for i in dot_T.index]
    dot_T.index.name = 'improve_sample_id'
    return dot_T

def indx_int_colm_flt(d_typ, dot_T):
    dot_T.columns = dot_T.columns.map(str)
    dot_T.columns = [i.split('.')[0] for i in dot_T.columns]
    dot_T.columns = ['entrz_' + d_typ + i for i in dot_T.columns]
    dot_T.columns.name = 'entrez_id'
    
    dot_T.index = dot_T.index.map(str)
    dot_T.index = ['smpl_id_' + i for i in dot_T.index]
    dot_T.index.name = 'improve_sample_id'
    return dot_T

def float_to_string(d_typ, dot_T):
    dot_T.columns = dot_T.columns.map(str)
    dot_T.columns = [i.split('.')[0] for i in dot_T.columns]
    dot_T.columns = ['entrz_' + d_typ + i for i in dot_T.columns]
    dot_T.columns.name = 'entrez_id'
    
    dot_T.index = dot_T.index.map(str)
    dot_T.index = [i.split('.')[0] for i in dot_T.index]
    dot_T.index = ['smpl_id_' + i for i in dot_T.index]
    dot_T.index.name = 'improve_sample_id'
    return dot_T