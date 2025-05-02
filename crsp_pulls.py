import math
import numpy as np
import pandas as pd
import time

from wrds_creds import read_pgpass
wrds_username,wrds_password=read_pgpass()
import wrds
db = wrds.Connection(wrds_host="wrds-pgdata.wharton.upenn.edu",
                     wrds_username=wrds_username,
                     wrds_password=wrds_password)


def test_query():
    test_df=db.raw_sql("""
        SELECT COUNT(*) 
        FROM crsp.dsf AS a
        INNER JOIN ff.factors_daily AS b
            ON a.date = b.date
        LEFT JOIN crsp.stocknames AS c
            ON a.permno = c.permno
            AND c.namedt <= (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date
            AND (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date <= c.nameenddt
            AND c.shrcd IN (10,11)
        WHERE a.date BETWEEN '2000-01-01' AND '2001-12-31'
                        """)
    return test_df

def get_endpoint_years():
    year_df=db.raw_sql("""
        SELECT max(extract(year from a.date)) as end_year,
                min(extract(year from a.date)) as start_year
        FROM crsp.dsf AS a
                        """)
    return year_df

# 1. Daily Data Pull
    # Merge CRSP dsf with FamaFrench daily to get daily excess returns 
    # and with CRSP stocknames to filter for common stocks
def daily_pull():

    daily_df=db.raw_sql(f"""
    select a.permno, a.date, a.ret, a.prc,
           extract(year from a.date) as y, 
            extract(month from a.date) as m, 
            extract(quarter from a.date) as q,
            (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date as mdate,
            (date_trunc('quarter', a.date) + interval '3 months' - interval '1 day')::date as qdate,
           b.mktrf, b.rf, a.ret-b.rf as exret,
            c.namedt, c.nameenddt, c.shrcd
    from crsp.dsf as a
    inner join ff.factors_daily as b
        on a.date = b.date
    left join crsp.stocknames as c
        on a.permno = c.permno
        and c.namedt <= (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date
        and (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date <= c.nameenddt 
        and c.shrcd in (10,11)
    fetch first 1000 rows only
    """, 
    date_cols=['date', 'mdate', 'qdate'],
    )
    #chunksize=500_000)
    #where a.date between '1964-07-01' and '2001-06-30' 
    return daily_df

# 2. Estimate Monthly Betas
    # exret~beta*mktrf
def estimate_beta(daily_data):

    #daily_data=daily_pull()
    daily_data = daily_data[daily_data['prc'].abs() > 5]
    #daily_data['permno'] = daily_data['permno'].astype(int)

    def fast_beta(group):
        if len(group) < 15:
            return pd.Series({'betamkt': np.nan})
        cov = np.cov(group['exret'], group['mktrf'])[0,1]
        var = np.var(group['mktrf'])
        beta = cov / var if var != 0 else np.nan
        return pd.Series({'betamkt': beta})

    betas = (
        daily_data.groupby(['permno', 'mdate'])
                .apply(fast_beta)
                .dropna()
                .reset_index()
    )
    return betas


def daily_pull_batch(start_year, end_year):
    """Pull a batch of CRSP daily data for specific years."""
    sql_query = f"""
    SELECT 
        a.permno, 
        a.date, 
        a.ret, 
        a.prc,
        EXTRACT(year FROM a.date) AS y, 
        EXTRACT(month FROM a.date) AS m, 
        EXTRACT(quarter FROM a.date) AS q,
        (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date AS mdate,
        (date_trunc('quarter', a.date) + interval '3 months' - interval '1 day')::date AS qdate,
        b.mktrf, 
        b.rf, 
        a.ret - b.rf AS exret,
        c.namedt, 
        c.nameenddt, 
        c.shrcd
    FROM crsp.dsf AS a
    INNER JOIN ff.factors_daily AS b
        ON a.date = b.date
    LEFT JOIN crsp.stocknames AS c
        ON a.permno = c.permno
        AND c.namedt <= (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date
        AND (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date <= c.nameenddt 
        AND c.shrcd IN (10,11)
    WHERE a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    FETCH FIRST 10000 ROWS ONLY
    """
    
    sql_query = f"""
    SELECT a.permno, a.date, a.ret, a.prc,
        EXTRACT(year FROM a.date) AS y, 
        EXTRACT(month FROM a.date) AS m, 
        EXTRACT(quarter FROM a.date) AS q,
        (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date AS mdate,
        (date_trunc('quarter', a.date) + interval '3 months' - interval '1 day')::date AS qdate,
        b.mktrf, b.rf, a.ret - b.rf AS exret
    FROM crsp.dsf AS a
    INNER JOIN ff.factors_daily AS b
        ON a.date = b.date
    WHERE a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    FETCH FIRST 10000 ROWS ONLY
    """
    print(sql_query)
    #FETCH FIRST 1000 ROWS ONLY
    # Pull with chunksize so no single memory blowup
    chunks = db.raw_sql(
        sql_query,
        date_cols=['date', 'mdate', 'qdate'],
        index_col=None,
        chunksize=500_000,   # 500k rows per chunk
    )

    batch_data = []
    for chunk in chunks:
        if isinstance(chunk, pd.DataFrame):
            batch_data.append(chunk)
        elif isinstance(chunk, str) and chunk.strip() in {'permno', 'date', 'ret', 'prc', 'y', 'm', 'q', 'mdate', 'qdate', 'mktrf', 'rf', 'exret', 'namedt', 'nameenddt', 'shrcd'}:
            # Just skip the header column names
            continue
        else:
            print(f"Unexpected chunk received: {chunk[:500]}")  # Only print if it's truly weird

    if batch_data:
        return pd.concat(batch_data, ignore_index=True)
    else:
        raise ValueError(f"No valid data pulled for {start_year}-{end_year}")
    
#version2
def daily_pull_batch_2(start_year, end_year):
    """Pull a batch of CRSP daily data for specific years."""
    sql_query = f"""
        SELECT 
            a.permno, a.date, a.ret, a.prc,
            EXTRACT(year FROM a.date) AS y, 
            EXTRACT(month FROM a.date) AS m, 
            EXTRACT(quarter FROM a.date) AS q,
            (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date AS mdate,
            (date_trunc('quarter', a.date) + interval '3 months' - interval '1 day')::date AS qdate,
            b.mktrf, b.rf, a.ret - b.rf AS exret,c.namedt, c.nameenddt, c.shrcd
        FROM crsp.dsf AS a
        INNER JOIN ff.factors_daily AS b
            ON a.date = b.date
        LEFT JOIN crsp.stocknames AS c
            ON a.permno = c.permno
            AND c.namedt <= (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date
            AND (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date <= c.nameenddt 
            AND c.shrcd IN (10,11)
        WHERE a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
        """
    #FETCH FIRST 10000 ROWS ONLY
    # Log the raw query output to check data type and size
    result = db.raw_sql(
        sql_query,
        date_cols=['date', 'mdate', 'qdate'],
        index_col=None  # Ensures the index is not set
    )
    return result
    """
    # Check what we are getting back (type and shape of the result)
    print(f"Type of result: {type(result)}")
    
    if isinstance(result, pd.DataFrame):
        print(f"First 5 rows of the result: \n{result.head()}")
        return result
    else:
        print("Result is not a DataFrame. Checking for other types...")
        if hasattr(result, 'fetchmany'):
            # If the result is a database cursor-like object
            print(f"Fetched {len(result.fetchmany(10))} rows.")
        else:
            print(f"Received a non-iterable result: {result}")
            
        # Check if chunks are returned as a generator and process them
        batch_data = []
        for chunk in result:
            if isinstance(chunk, pd.DataFrame):
                batch_data.append(chunk)
            else:
                print(f"Unexpected chunk received: {chunk}")
        
        if batch_data:
            return pd.concat(batch_data, ignore_index=True)
        else:
            print("No valid data pulled.")
            raise ValueError(f"No valid data pulled for {start_year}-{end_year}")
    """


def aggregate_crsp_batches(start_year,end_year):

    all_data = []
    # Define 5-year intervals
    interval_length=2
    years = list(range(start_year, end_year, interval_length))  # [1975, 1980, 1985, ..., 2020]
    for batch_year in years:
        batch_end = batch_year + (interval_length-1)
        print(f"Pulling {batch_year}-{batch_end}")
        #batch = daily_pull_batch(batch_year, batch_end)
        #all_data.append(batch)

        #batch = daily_pull_batch(batch_year, batch_end)
        batch = daily_pull_batch_2(batch_year, batch_end)
        time.sleep(2)
        
        all_data.append(batch)
        #batch.to_parquet(f"crsp_dsf_{batch_year}_{batch_end}.parquet")
        #print(f"Saved crsp_dsf_{batch_year}_{batch_end}.parquet")

    # Combine all batches
    full_daily_data = pd.concat(all_data, ignore_index=True)
    return full_daily_data

def get_monthly_betas(start_year,end_year):
    #make this like the above
    print('pulling CRSP data')
    full_daily_data = aggregate_crsp_batches(start_year,end_year)
    print('Estimating Betas')
    betas = estimate_beta(full_daily_data)
    return betas
#    return None


def monthly_pull():

    monthly_df = db.raw_sql("""
    select a.permno, a.date, a.ret, a.prc, a.shrout,
           year(a.date) as y, month(a.date) as m, qtr(a.date) as q,
           intnx('month', a.date, 0, 'E') as mdate format=date9.,
           intnx('quarter', a.date, 0, 'E') as qdate format=date9.,
           intnx('month', a.date, -1, 'E') as mdate1 format=date9.,
           intnx('quarter', a.date, -1, 'E') as qdate1 format=date9.
    from crsp.msf as a""", 
    date_cols=['date', 'mdate', 'qdate', 'mdate1', 'qdate1'])
    monthly_data['mktcap'] = (monthly_data['prc'].abs()) * monthly_data['shrout']
    # Sort and create lagged market cap
    monthly_data = monthly_data.sort_values(['permno', 'date'])
    monthly_data['lmktcap'] = monthly_data.groupby('permno')['mktcap'].shift(1)
    return monthly_df

def ff_monthly_pull():
    ff_monthly = db.raw_sql("""
        select date, mktrf, smb, hml, umd, rf
        from ff.factors_monthly""", 
        date_cols=['date'])
    ff_monthly['date'] = ff_monthly['date'] + pd.offsets.MonthEnd(0)
    return ff_monthly


def monthly_pull_batch(start_year, end_year):
    sql_query = f"""
    SELECT 
        a.permno, a.date, a.ret, a.prc, a.shrout, abs(a.prc)*a.shrout as mktcap,
        EXTRACT(year FROM a.date) AS y, 
        EXTRACT(month FROM a.date) AS m, 
        EXTRACT(quarter FROM a.date) AS q,
        (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date AS mdate,
        (date_trunc('quarter', a.date) + interval '3 months' - interval '1 day')::date AS qdate,
        (date_trunc('month', a.date) - interval '1 day')::date AS mdate1,
        (date_trunc('quarter', a.date) - interval '1 day')::date AS qdate1,
        LAG(ABS(a.prc) * a.shrout) OVER (PARTITION BY a.permno ORDER BY a.date) AS lmktcap,
        b.mktrf, b.smb, b.hml, b.umd, b.rf, a.ret - b.rf AS exret,
        c.namedt, c.nameenddt, c.shrcd
    FROM crsp.dsf AS a
    INNER JOIN ff.factors_daily AS b
        ON a.date = b.date
    LEFT JOIN crsp.stocknames AS c
        ON a.permno = c.permno
        AND c.namedt <= (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date
        AND (date_trunc('month', a.date) + interval '1 month' - interval '1 day')::date <= c.nameenddt 
        AND c.shrcd IN (10,11)
    WHERE a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    """
    # Pull with chunksize so no single memory blowup
    #chunks = db.raw_sql(
    #    sql_query,
    #    date_cols=['date', 'mdate', 'qdate', 'mdate1', 'qdate1'],
    #    chunksize=500_000,   # 500k rows per chunk
    #)
    result = db.raw_sql(sql_query,date_cols=['date', 'mdate', 'qdate'],index_col=None)
    return result
    """
    batch_data = []
    for chunk in chunks:
        if isinstance(chunk, pd.DataFrame):   
            batch_data.append(chunk)
        else:
            #print(f"Skipping non-DataFrame chunk: {type(chunk)}")
            print(f"Non-DataFrame chunk received: {chunk[:500]}")  # print first 500 characters
    
    if batch_data:
        return pd.concat(batch_data, ignore_index=True)
    else:
        raise ValueError(f"No valid data pulled for {start_year}-{end_year}")
    """

def aggregate_monthly_crsp_batches(start_year,end_year):

    all_data = []
    # Define 5-year intervals
    interval_length=2
    years = list(range(start_year, end_year, interval_length))  # [1975, 1980, 1985, ..., 2020]
    for batch_year in years:
        batch_end = batch_year + (interval_length-1)
        print(f"Pulling {batch_year}-{batch_end}")
        #batch = daily_pull_batch(batch_year, batch_end)
        #all_data.append(batch)

        batch = monthly_pull_batch(batch_year, batch_end)
        time.sleep(2)
        all_data.append(batch)
        #batch.to_parquet(f"crsp_dsf_{batch_year}_{batch_end}.parquet")
        #print(f"Saved crsp_dsf_{batch_year}_{batch_end}.parquet")

    # Combine all batches
    full_monthly_data = pd.concat(all_data, ignore_index=True)
    return full_monthly_data

#6. Merge Lagged Betas with Next Month Return
def merge_monthly_tables(crsp,betas):
    rets = crsp.merge(
        betas[['permno', 'mdate', 'betamkt']],
        left_on=['permno', 'mdate1'],
        right_on=['permno', 'mdate'],
        how='inner'
    )
    rets = rets[rets['ret'] > -2]
    rets = rets[(rets['lmktcap'] > 0) & (~rets['betamkt'].isna())]
    return rets

#7. Double Sort: Size × Beta
#8. Collapse by Month (Weighted Average Returns)
#9. Merge in Fama-French Monthly Factors
#10. Create Special Portfolios (like temp1 in SAS)
#11. Final Dataset


###############################################
###############################################


# 2. Monthly Data Pull (MSF + lagging already built)
def monthly_pull():

    monthly_df = db.raw_sql("""
    select a.permno, a.date, a.ret, a.prc, a.shrout,
           year(a.date) as y, month(a.date) as m, qtr(a.date) as q,
           intnx('month', a.date, 0, 'E') as mdate format=date9.,
           intnx('quarter', a.date, 0, 'E') as qdate format=date9.,
           intnx('month', a.date, -1, 'E') as mdate1 format=date9.,
           intnx('quarter', a.date, -1, 'E') as qdate1 format=date9.
    from crsp.msf as a""", 
    date_cols=['date', 'mdate', 'qdate', 'mdate1', 'qdate1'])
    monthly_data['mktcap'] = (monthly_data['prc'].abs()) * monthly_data['shrout']
    # Sort and create lagged market cap
    monthly_data = monthly_data.sort_values(['permno', 'date'])
    monthly_data['lmktcap'] = monthly_data.groupby('permno')['mktcap'].shift(1)
    return monthly_df

# 3. Stocknames Pull
def stocknames_pull():

    stocknames = db.raw_sql("""
    select permno, namedt, nameenddt, shrcd
    from crsp.stocknames""", 
    date_cols=['namedt', 'nameenddt'])
    return stocknames

# 4. Fama-French Monthly Factors Pull
def ff_monthly_pull():
    ff_monthly = db.raw_sql("""
        select date, mktrf, smb, hml, umd, rf
        from ff.factors_monthly""", 
        date_cols=['date'])
    ff_monthly['date'] = ff_monthly['date'] + pd.offsets.MonthEnd(0)
    return ff_monthly


def ff_monthly_pull2():
    ff_monthly = db.raw_sql("""
        select *
        from ff.factors_daily
        fetch first 10 rows only                    
        """, 
        date_cols=['date'])
    ff_monthly['date'] = ff_monthly['date'] + pd.offsets.MonthEnd(0)
    return ff_monthly

"""
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
# Assume you already have daily_data pulled earlier
# Step 1: Filter for valid observations
daily_data = daily_data[daily_data['prc'].abs() > 5]
# Step 2: Create keys
daily_data['permno'] = daily_data['permno'].astype(int)
daily_data['mdate'] = pd.to_datetime(daily_data['mdate'])
# Step 3: Set up panel
daily_data = daily_data.set_index(['permno', 'date'])
# Step 4: Create "excess return"
daily_data['exret'] = daily_data['ret'] - daily_data['rf']
# Step 5: Collapse into monthly groups
daily_data['month_id'] = daily_data.index.get_level_values('date').to_period('M')
daily_data['group_id'] = daily_data.index.get_level_values('permno').astype(str) + "_" + daily_data['month_id'].astype(str)
# Step 6: Define function to estimate beta for each (permno, month)
def estimate_beta(group):
    if len(group) < 15:   # must have at least 15 daily observations
        return pd.Series({'betamkt': np.nan})
    y = group['exret']
    X = group[['mktrf']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return pd.Series({'betamkt': model.params['mktrf']})
# Step 7: Apply function groupby
betas = (
    daily_data
    .reset_index()
    .groupby(['permno', 'mdate'])
    .apply(estimate_beta)
    .dropna()
    .reset_index()
)
betas.head()

# skip statsmodels altogether and use a vectorized closed-form OLS:
def fast_beta(group):
    if len(group) < 15:
        return pd.Series({'betamkt': np.nan})
    cov = np.cov(group['exret'], group['mktrf'])[0,1]
    var = np.var(group['mktrf'])
    beta = cov / var if var != 0 else np.nan
    return pd.Series({'betamkt': beta})
"""

# After Pulling:
#   You have daily returns + FF factors ready for beta estimation.
#   You have monthly returns + lagged mkt cap ready for forming portfolios.
#   You have monthly FF factors ready for adjusting returns.
#   You have stocknames for filtering common stock (shrcd 10,11).

##########################################################################################
# 1. Pull Daily CRSP (DSF) and add date variables
#dsf = db.raw_sql("""
#    select permno, date, ret, prc
#    from crsp.dsf
#""")
#dsf['date'] = pd.to_datetime(dsf['date'])
#dsf['y'] = dsf['date'].dt.year
#dsf['m'] = dsf['date'].dt.month
#dsf['q'] = dsf['date'].dt.quarter
#dsf['mdate'] = dsf['date'] + pd.offsets.MonthEnd(0)
#dsf['qdate'] = (dsf['date'] + pd.offsets.MonthEnd(0)).apply(lambda d: d + pd.offsets.MonthEnd((3 - d.month%3)%3))

# 2. Merge with Fama-French daily factors
#ff_factors = db.raw_sql("""
#    select date, mktrf, rf
#    from ff.factors_daily
#""")
#ff_factors['date'] = pd.to_datetime(ff_factors['date'])
#dsf = dsf.merge(ff_factors, on='date', how='inner')
#dsf['exret'] = dsf['ret'] - dsf['rf']

# 3. Estimate pre-ranking betas by permno × month
# ~We run monthly regressions of exret ~ mktrf for each stock.
#import statsmodels.api as sm
#def regress_beta(x):
#    if len(x) >= 15:
#        y = x['exret']
#        X = sm.add_constant(x['mktrf'])
#        model = sm.OLS(y, X).fit()
#        return pd.Series({'betamkt': model.params['mktrf'], 'n_obs': int(model.df_resid + model.df_model + 1)})
#    else:
#        return pd.Series({'betamkt': np.nan, 'n_obs': np.nan})
#dsf_grouped = dsf[dsf['prc'] > 5].groupby(['permno', 'mdate'])
#betas = dsf_grouped.apply(regress_beta).dropna()
#betas = betas.reset_index()

# 4. Restrict to Common Stocks (10,11)
#stocknames = db.raw_sql("""
#    select permno, namedt, nameenddt, shrcd
#    from crsp.stocknames
#""")
#stocknames['namedt'] = pd.to_datetime(stocknames['namedt'])
#stocknames['nameenddt'] = pd.to_datetime(stocknames['nameenddt'])
# Merge
#betas = betas.merge(stocknames, on='permno', how='left')
#betas = betas[(betas['shrcd'].isin([10,11])) &
#              (betas['mdate'] >= betas['namedt']) & 
#              (betas['mdate'] <= betas['nameenddt'])]

# 5. Pull Monthly CRSP (MSF) and add date variables
#msf = db.raw_sql("""
#    select permno, date, ret, prc, shrout
#    from crsp.msf
#""")
#msf['date'] = pd.to_datetime(msf['date'])
#msf['y'] = msf['date'].dt.year
#msf['m'] = msf['date'].dt.month
#msf['q'] = msf['date'].dt.quarter
#msf['mdate'] = msf['date'] + pd.offsets.MonthEnd(0)
#msf['qdate'] = (msf['date'] + pd.offsets.MonthEnd(0)).apply(lambda d: d + pd.offsets.MonthEnd((3 - d.month%3)%3))
#msf['mdate1'] = msf['date'] + pd.offsets.MonthEnd(-1)
#msf['qdate1'] = msf['date'] + pd.offsets.MonthEnd(-3)
#msf['mktcap'] = (msf['prc'].abs()) * msf['shrout']
#Lag market cap:
#msf = msf.sort_values(['permno', 'date'])
#msf['lmktcap'] = msf.groupby('permno')['mktcap'].shift(1)

# 6. Merge Lagged Betas with Next Month Return
#rets = msf.merge(
#    betas[['permno', 'mdate', 'betamkt']],
#    left_on=['permno', 'mdate1'],
#    right_on=['permno', 'mdate'],
#    how='inner'
#)
#rets = rets[rets['ret'] > -2]
#rets = rets[(rets['lmktcap'] > 0) & (~rets['betamkt'].isna())]

# 7. Double Sort: Size × Beta
#First, sort by date:
#rets = rets.sort_values(['date'])
#Rank into 5 groups by lmktcap:
#rets['sizerank'] = rets.groupby('date')['lmktcap'].transform(
#    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
#)
#Then sort within size groups by betamkt:
#rets = rets.sort_values(['date', 'sizerank'])
#rets['betarank'] = rets.groupby(['date', 'sizerank'])['betamkt'].transform(
#    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
#)
#Add 1 to match SAS numbering and create portfolio labels:
#rets['sizerank'] = rets['sizerank'] + 1
#rets['betarank'] = rets['betarank'] + 1
#rets['port'] = 'S' + rets['sizerank'].astype(str) + 'B' + rets['betarank'].astype(str)

# 8. Collapse by Month (Weighted Average Returns)
#portrets = rets.groupby(['date', 'port', 'sizerank', 'betarank']).apply(
#    lambda x: np.average(x['ret'], weights=x['lmktcap'])
#).reset_index(name='ret')

# 9. Merge in Fama-French Monthly Factors
#ff_monthly = db.raw_sql("""
#    select date, mktrf, smb, hml, rf
#    from ff.factors_monthly
#""")
#ff_monthly['date'] = pd.to_datetime(ff_monthly['date']) + pd.offsets.MonthEnd(0)
#portrets = portrets.merge(ff_monthly, on='date', how='left')
#portrets['exret'] = portrets['ret'] - portrets['rf']
#portrets['y'] = portrets['date'].dt.year

#10. Create Special Portfolios (like temp1 in SAS)
#Subset to betarank in (1,5):
#temp1 = portrets[portrets['betarank'].isin([1,5])]
#temp1 = temp1.sort_values(['date', 'sizerank', 'betarank'])
#temp1['lret'] = temp1.groupby(['date', 'sizerank'])['exret'].shift(1)
# Adjust exret and redefine port
#temp1 = temp1.dropna(subset=['lret'])
#temp1['exret'] = temp1['exret'] - temp1['lret']
#temp1['betarank'] = 0
#temp1['port'] = 'S' + temp1['sizerank'].astype(str) + 'B0'
#temp1 = temp1.drop(columns=['lret'])

# 11. Final Dataset
#portrets_final = pd.concat([portrets, temp1], ignore_index=True)
#portrets_final = portrets_final.sort_values(['betarank', 'date'])
