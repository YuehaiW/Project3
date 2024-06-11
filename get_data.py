import pymysql
import numpy as np
import pandas as pd

from datetime import datetime


def get_adjust_dt(date_ls):
    """
    获取每月的最后一个交易日
    """
    result = []
    for i in range(len(date_ls) - 1):
        if date_ls[i].month != date_ls[i + 1].month:
            result.append(date_ls[i])
    if result[-1] != date_ls[-1]:
        result.append(date_ls[-1])
    result = [datetime.strftime(date, '%Y%m%d') for date in result]
    return result


def month_end(date):
    """
    获取某个日期所在月份的最后一天的日期。
    如果输入日期正好是月末,则返回同一个日期。
    """
    end_of_month = date + pd.offsets.MonthEnd()
    if date.month == end_of_month.month:
        return end_of_month
    else:
        return date


def fill_report_period(df, end):
    """
    对于按季度公布的部分财务数据，填充上个公布日到下个公布日之间的月度数据
    """
    codes = df.code.unique()
    df_expand = pd.DataFrame()

    for code in codes:
        tmp = df[df.code == code].copy()
        end_date = min(tmp['date'].max() + pd.DateOffset(months=3), pd.to_datetime(end))
        all_month_ends = pd.date_range(start=tmp['date'].min(), end=end_date,
                                       freq='M', inclusive='right')
        tmp = tmp.set_index('date').reindex(all_month_ends, method='ffill').reset_index()
        tmp = tmp.rename(columns={'index': 'date'})
        df_expand = pd.concat([df_expand, tmp], axis=0)

    return df_expand


def get_trade(start, end):
    """获取与收益率，收盘价等相关的因子"""
    # 获取数据
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    query = """
            select TRADE_DT, S_INFO_WINDCODE, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_PCTCHANGE
            from ASHAREEODPRICES
            where TRADE_DT between '{start}' and '{end}'
            order by TRADE_DT, S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code', 'close', 'volume', 'daily_return'])
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)
    df1 = df[['date', 'code', 'close']].pivot(index='date', columns='code', values='close')
    df2 = df[['date', 'code', 'volume']].pivot(index='date', columns='code', values='volume')

    # 计算120天成交量，最低、平均收盘价
    close_20d_min = df1.rolling(window=20).min()
    close_20d_mean = df1.rolling(window=20).mean()
    volume_120d_sum = df2.rolling(window=120).sum()
    close_20d_min = pd.melt(close_20d_min.reset_index(), id_vars='date', var_name='code', value_name='close_20d_min')
    close_20d_mean = pd.melt(close_20d_mean.reset_index(), id_vars='date', var_name='code', value_name='close_20d_mean')
    volume_120d_sum = pd.melt(volume_120d_sum.reset_index(), id_vars='date', var_name='code',
                              value_name='volume_120d_sum')
    trade_df = pd.merge(close_20d_min, close_20d_mean, on=['date', 'code']).merge(volume_120d_sum, on=['date', 'code'])

    # 计算近1月、6月收益率
    yieldm = df1 / df1.shift(21) - 1
    yield6m = df1 / df1.shift(126) - 1
    yieldm = pd.melt(yieldm.reset_index(), id_vars='date', var_name='code', value_name='yieldm')
    yield6m = pd.melt(yield6m.reset_index(), id_vars='date', var_name='code', value_name='yield6m')
    trade_df = pd.merge(trade_df, yieldm, on=['date', 'code']).merge(yield6m, on=['date', 'code'])
    # 计算52周波动率
    df3 = df[['date', 'code', 'daily_return']].pivot(index='date', columns='code', values='daily_return')
    volatility = df3.rolling(window=260).std()
    volatility = pd.melt(volatility.reset_index(), id_vars='date', var_name='code', value_name='volatility')
    trade_df = pd.merge(trade_df, volatility, on=['date', 'code'])
    dates = pd.to_datetime(df.date.unique())
    adj_dates = get_adjust_dt(dates)
    fac_trade = trade_df[trade_df.date.isin(adj_dates)].copy()
    fac_trade['date'] = pd.to_datetime(fac_trade.date).apply(month_end)

    return fac_trade


def get_basic(start, end):
    """获取市值，换手率等交易基本信息"""
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    query = """
            select TRADE_DT, S_INFO_WINDCODE, S_VAL_MV, S_VAL_PB_NEW, S_VAL_PE_TTM, S_VAL_PS_TTM, S_DQ_TURN
            from ASHAREEODDERIVATIVEINDICATOR
            where TRADE_DT between '{start}' and '{end}'
            order by TRADE_DT, S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code', 'cap', 'pb', 'pe', 'ps', 'turnrate'])
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)

    df.dropna(subset=['turnrate'], inplace=True, axis=0)  # 保留交易日
    basic = df.drop('turnrate', axis=1)
    dates = pd.to_datetime(basic.date.unique())
    adj_dt = get_adjust_dt(dates)
    basic = basic[basic.date.isin(adj_dt)]

    # 计算近1月，6月换手率
    tmp = df[['date', 'code', 'turnrate']]
    turnover = tmp.pivot(index='date', columns='code', values='turnrate')
    turnratem = turnover.rolling(21).mean()
    turnrate6m = turnover.rolling(126).mean()

    turnratem = turnratem.loc[adj_dt]
    turnrate6m = turnrate6m.loc[adj_dt]

    # 合并数据为一个数据框
    res1 = pd.melt(turnratem.reset_index(), id_vars='date', var_name='code', value_name='turnratem')
    res2 = pd.melt(turnrate6m.reset_index(), id_vars='date', var_name='code', value_name='turnrate6m')
    res = pd.merge(res1, res2, on=['date', 'code']).merge(basic, on=['date', 'code'])
    res.reset_index(drop=True, inplace=True)
    df_basic = res.copy()
    df_basic.date = pd.to_datetime(df_basic.date).apply(month_end)

    return df_basic


def get_finance(start, end):
    """获得净利润扣非前后孰低值"""
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    query = """
            select ACTUAL_ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, NET_PROFIT_INCL_MIN_INT_INC, NET_PROFIT_AFTER_DED_NR_LP 
            from ASHAREINCOME
            where ACTUAL_ANN_DT between '{start}' and '{end}' and STATEMENT_TYPE=408001000
            order by ACTUAL_ANN_DT, S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code', 'report', 'profit1', 'profit2'])
    df[['profit1', 'profit2']] = df[['profit1', 'profit2']].astype(float)
    df.dropna(subset=['profit1', 'profit2'], how='all', inplace=True)  # 两列都为空值则删除
    df['net_profit'] = np.nanmin(df[['profit1', 'profit2']], axis=1)
    df_finance = df[['date', 'code', 'report', 'net_profit']].copy()
    df_finance['date'] = pd.to_datetime(df_finance['date']).apply(month_end)
    df_finance['report'] = pd.to_datetime(df_finance['report'])
    # 可能一个公布日会公布前两个季度的信息，按报告期靠后的季度计算
    df_finance = df_finance.loc[df_finance.groupby(['date', 'code'])['report'].idxmax()]
    df_finance.drop(columns='report', inplace=True)
    df_finance = fill_report_period(df_finance, end)

    return df_finance


def get_profit(start, end):
    """获得盈利相关的因子"""
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    query = """
            select ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, S_FA_EPS_BASIC, S_FA_GROSSPROFITMARGIN, S_FA_NETPROFITMARGIN, 
                   S_FA_ROE, S_FA_ROA, S_FA_ARTURN, S_FA_FATURN, S_FA_ASSETSTURN, S_FA_YOY_OR, S_QFA_CGRSALES, 
                   S_FA_YOYEPS_BASIC, S_QFA_YOYNETPROFIT 
            from ASHAREFINANCIALINDICATOR
            where ANN_DT between '{start}' and '{end}'
            order by ANN_DT, S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['date', 'code', 'report', 'eps', 'gross_margin', 'net_margin', 'roe', 'roa',
                                     'arturn', 'faturn', 'assetsturn', 'or_yoy', 'or_qoq', 'eps_yoy', 'net_profit_yoy'])
    df.iloc[:, 3:] = df.iloc[:, 3:].astype(float)
    df['date'] = pd.to_datetime(df['date']).apply(month_end)
    df['report'] = pd.to_datetime(df['report'])
    df = df.loc[df.groupby(['date', 'code'])['report'].idxmax()]
    df_profit = df.drop(columns='report')
    df_profit = fill_report_period(df_profit, end)

    return df_profit


def get_ctdown(start, end):
    """获得近一月连续下降天数和幅度"""
    def get_ctdowndc(ser):
        max_count = s = 0
        max_chg, p = 0, 1
        if ser.isna().all():
            return pd.NA
        else:
            ser = ser.dropna()
            for i in range(len(ser)):
                if ser[i] < 0:
                    s += 1
                    max_count = max(max_count, s)
                    p = p * (1 + ser[i])
                    max_chg = min(max_chg, p - 1)
                else:
                    s = 0
                    p = 1
            return [max_count, abs(max_chg)]

    # 获取一个自然月内第一个和最后一个交易日
    def get_month(date_ls):
        result = {'start': [], 'end': []}
        for i in range(len(date_ls) - 1):
            if date_ls[i].month != date_ls[i + 1].month:
                result['start'].append(date_ls[i + 1])
                result['end'].append(date_ls[i])

        if result['end'][-1] != date_ls[-1]:
            result['end'].append(date_ls[-1])
        if result['start'][0] != date_ls[0]:
            result['start'].insert(0, date_ls[0])

        ym = [date.strftime('%Y-%m') for date in result['start']]
        result = pd.DataFrame(result, index=ym)
        return result

    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    query = """
            select TRADE_DT, S_INFO_WINDCODE, S_DQ_PCTCHANGE
            from ASHAREEODPRICES
            where TRADE_DT between '{start}' and '{end}'
            order by TRADE_DT, S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df_rets = pd.DataFrame(data, columns=['date', 'code', 'daily_return'])
    df_rets['daily_return'] = df_rets['daily_return'].astype(float) / 100
    dates = get_month(pd.to_datetime(df_rets.date.unique()))
    df_rets = df_rets.pivot(index='date', columns='code', values='daily_return')

    continuous_day = pd.DataFrame(pd.NA, index=dates['end'], columns=df_rets.columns)
    continuous_chg = pd.DataFrame(pd.NA, index=dates['end'], columns=df_rets.columns)
    # 每次循环取一自然月数据，计算连续下降天数与幅度
    for i in range(len(dates)):
        start, end = dates.iloc[i, 0], dates.iloc[i, 1]
        start, end = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')
        tmp = df_rets.loc[start:end, :]
        ctd_res = tmp.apply(get_ctdowndc)
        continuous_day.iloc[i, :] = ctd_res.iloc[0, :]
        continuous_chg.iloc[i, :] = ctd_res.iloc[1, :]
    continuous_day.index.name = 'date'
    continuous_chg.index.name = 'date'
    ctday = pd.melt(continuous_day.reset_index(), id_vars='date', var_name='code', value_name='ctdowndays')
    ctchg = pd.melt(continuous_chg.reset_index(), id_vars='date', var_name='code', value_name='ctdownchg')
    ctday = ctday.dropna()
    ctchg = ctchg.dropna()
    con_res = pd.merge(ctday, ctchg, on=['date', 'code'])
    con_res['date'] = pd.to_datetime(con_res['date']).apply(month_end)

    return con_res


def get_shrcr(start, end):
    """获取第一大股东持股比例"""
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()

    # 部分股票一月公布两次持股比例，因此分月份取持股最高的比例
    query = """
            select year(ANN_DT), month(ANN_DT), S_INFO_WINDCODE, MAX(S_HOLDER_PCT) as max_pct
            from ASHAREINSIDEHOLDER
            where ANN_DT between '{start}' and '{end}'
            group by S_INFO_WINDCODE, year(ANN_DT), month(ANN_DT)
            order by year(ANN_DT), month(ANN_DT), S_INFO_WINDCODE
            """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['y', 'm', 'code', 'shrcr'])
    df['shrcr'] = df['shrcr'].astype(float)
    # 将年月转换为日期
    df['date'] = pd.to_datetime(df[['y', 'm']].apply(lambda x: pd.Timestamp(year=x['y'], month=x['m'], day=1), axis=1))
    df.drop(columns=['y', 'm'], inplace=True)
    df['date'] = df['date'].apply(month_end)

    df_holder = fill_report_period(df, end)
    return df_holder


def get_st(start, end):
    """获取股票的ST信息，由于要以当月因子值预测下月ST股，此处将ST月份向推一个月以与因子值日期对齐"""
    conn = pymysql.connect(
        host="192.168.7.93",
        user="quantchina",
        password="zMxq7VNYJljTFIQ8",
        database="wind",
        charset="gbk"
    )
    cursor = conn.cursor()
    query = """
        select a.TRADE_DT, a.S_INFO_WINDCODE
        from (select TRADE_DT, S_INFO_WINDCODE
              from ASHAREEODPRICES
              where TRADE_DT between '{start}' and '{end}') a
        join (select S_INFO_WINDCODE, ENTRY_DT, REMOVE_DT
              from ASHAREST
              where ENTRY_DT<='{end}' and (REMOVE_DT>='{start}' or REMOVE_DT is null)) b
        on a.S_INFO_WINDCODE=b.S_INFO_WINDCODE and a.TRADE_DT>=b.ENTRY_DT and (a.TRADE_DT<b.REMOVE_DT or 
           b.REMOVE_DT is null)
        order by a.TRADE_DT, a.S_INFO_WINDCODE
        """.format(start=start, end=end)
    cursor.execute(query)
    data = cursor.fetchall()
    df_st = pd.DataFrame(data, columns=['st_date', 'st_code'])
    st_df = pd.DataFrame(columns=['date', 'code', 'is_st'])
    # 将ST股记为1
    for code in df_st.st_code.unique():
        tmp = df_st[df_st.st_code == code].copy()
        st_date = (pd.to_datetime(tmp['st_date']).apply(month_end)).unique()
        st_code = pd.DataFrame({'date': st_date,
                                'code': [code] * len(st_date),
                                'is_st': [1] * len(st_date)})
        st_df = pd.concat([st_df, st_code], ignore_index=True)
    # 将ST的日期变换到上月以便与解释变量对齐
    new_date = []
    for i in range(len(st_df.date)):
        dt = st_df.date[i]
        new_date.append(dt - pd.DateOffset(months=1))
    st_df.date = new_date
    st_df['date'] = st_df['date'].apply(month_end)

    return st_df


if __name__ == '__main__':
    df_trade = get_trade('20081101', '20240430')
    df_basic = get_basic('20090101', '20240430')
    df_finance = get_finance('20090101', '20240430')
    df_profit = get_profit('20090101', '20240430')
    df_ctdown = get_ctdown('20100101', '20240430')
    df_h1 = get_shrcr('20090101', '20240430')
    st_df = get_st('20100101', '20240531')

    df_merge = pd.merge(df_trade, df_basic, on=['date', 'code']).merge(df_ctdown, on=['date', 'code'])
    raw_data = pd.merge(df_finance, df_profit, on=['date', 'code']).merge(df_merge, on=['date', 'code'])
    raw_data = pd.merge(raw_data, df_h1, on=['date', 'code'])
    res = pd.merge(raw_data, st_df, on=['date', 'code'], how='left')
    res['is_st'].fillna(0, inplace=True)  # 合并后非ST股记为0
    res_df = res[res['date'] >= '20100101']
    res_df.to_parquet('ST data.parquet')
