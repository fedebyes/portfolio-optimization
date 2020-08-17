import time

import numpy as np
import numpy.random as nrand
import pandas as pd


def main_loop(num_portfolios):
    global len_stocks
    stocks_prices = pd.read_csv('../data/DEFINITIVE/prices_from2008_csv.csv', parse_dates=["Date"], index_col="Date")
    stocks_mc = pd.read_csv('../data/DEFINITIVE/MC_from2008_csv.csv', parse_dates=["Date"], index_col="Date")
    stocks_pb = pd.read_csv('../data/DEFINITIVE/PTB_from2008_csv.csv', parse_dates=["Date"], index_col="Date")
    stocks_s_c = pd.read_csv('../data/DEFINITIVE/categories_csv.csv', index_col="sector_country")
    stocks_s_c = stocks_s_c.transpose()

    returns = stocks_prices.pct_change().dropna()
    returns.head()

    volatility = returns.rolling(window=60).std() * np.sqrt(12)
    stock_vol = volatility.dropna()
    countries = stocks_s_c["Country "].unique()
    stocks_by_country = {}

    for country in countries:
        stocks_by_country[country] = stocks_s_c[stocks_s_c["Country "] == country].index.values.tolist()

    # Create dictionary of stocks by sector
    sectors = stocks_s_c["Sector name "].unique()
    stocks_by_sector = {}
    for sector in sectors:
        stocks_by_sector[sector] = stocks_s_c[stocks_s_c["Sector name "] == sector].index.values.tolist()

    # dates
    dates = pd.DatetimeIndex(stocks_mc.index)
    years = dates.year.unique()
    months = dates.month.unique()

    # Create dictionary of stocks by market-cap quintile
    mean_by_month = stocks_mc
    df_mc_quint = {}

    for year in years:
        temp_year = mean_by_month[mean_by_month.index.year == year]
        for month in months:

            temp_month = temp_year[temp_year.index.month == month]
            if (len(temp_month) > 0):
                df_mc_quint[str(year) + "-" + str(month)] = pd.qcut(temp_month.squeeze(), 5, labels=range(5))

    df_mc_quint = pd.DataFrame.from_dict(df_mc_quint)

    # Create dictionary of stocks by price-book quintile
    pb_by_month = stocks_pb
    df_pb_quint = {}
    for year in years:
        temp_year = pb_by_month[pb_by_month.index.year == year]
        for month in months:
            temp_month = temp_year[temp_year.index.month == month]
            if (len(temp_month) > 0):
                df_pb_quint[str(year) + "-" + str(month)] = pd.qcut(temp_month.squeeze(), 5, labels=range(5))

    df_pb_quint = pd.DataFrame.from_dict(df_pb_quint)

    dates = pd.DatetimeIndex(stock_vol.index)
    years = dates.year.unique()
    volatility_by_month = {}

    # volatility per stock per period

    for year in years:
        temp_year = stock_vol[stock_vol.index.year == year]
        for month in months:
            temp_month = temp_year[temp_year.index.month == month]
            if (len(temp_month) > 0):
                volatility_by_month[str(year) + "-" + str(month)] = temp_month.reset_index(drop=True).loc[0]

    volatility_by_month_df = pd.DataFrame.from_dict(volatility_by_month)
    volatility_by_month_df.head()

    stocks_names = stocks_mc.columns.values.tolist()
    start = time.time()
    porfolios_entropy_sector = {}
    porfolios_entropy_country = {}
    porfolios_entropy_mc = {}
    porfolios_entropy_pb = {}
    stock_all_weights = {}

    # Main Execution
    for portfolio in range(num_portfolios):
        start_portfolio = time.time()
        print("Portfolio {} ...".format(portfolio))
        num_stocks = len(stocks_s_c)

        # random weights-monte carlo simulation
        weights = np.random.random(num_stocks)
        weights = weights / np.sum(weights)

        # Store weights for each portfolio
        stock_weights = {}
        for i, stock in enumerate(stocks_names):
            stock_weights[stock] = weights[i]

        stock_all_weights[portfolio] = stock_weights

        # entropy sectors dimension
        dimension_sector = {}

        for year in years:
            for month in months:
                pij = 0
                for sector in stocks_by_sector:
                    stocks = stocks_by_sector[sector]
                    len_stocks = len(stocks)
                    for stock in stocks:
                        key = str(year) + "-" + str(month)
                        if (key in volatility_by_month_df.columns):
                            pij += (volatility_by_month[key][stock] * stock_weights[stock]) / len_stocks
                pij = pij * np.log(np.nan_to_num(pij))
                dimension_sector[key] = pij
        porfolios_entropy_sector[portfolio] = dimension_sector
        stop_sector = time.time()
        print("Total time sector = " + str(float(stop_sector - start_portfolio)) + " seconds")

        # entropy country dimension
        dimension_country = {}
        for year in years:
            for month in months:
                pij = 0
                for country in stocks_by_country:
                    stocks = stocks_by_country[country]
                    len_stocks = len(stocks)
                    for stock in stocks:
                        key = str(year) + "-" + str(month)
                        if (key in volatility_by_month):
                            pij += (volatility_by_month[key][stock] * stock_weights[stock]) / len_stocks
                pij = pij * np.log(np.nan_to_num(pij))
                dimension_country[key] = pij
        porfolios_entropy_country[portfolio] = dimension_country
        stop_country = time.time()
        print("Total time country= " + str(float(stop_country - stop_sector)) + " seconds")

        # entropy market-cap dimension
        dimension_mc = {}
        for year in years:
            for month in months:
                pij = 0
                for quintil in range(5):
                    key = str(year) + "-" + str(month)
                    stocks = df_mc_quint[key][df_mc_quint[key] == quintil].index
                    len_stocks = len(stocks)
                    for stock in stocks:
                        if (key in volatility_by_month):
                            pij += (volatility_by_month[key][stock] * stock_weights[stock]) / len_stocks
                pij = pij * np.log(np.nan_to_num(pij))
                dimension_mc[key] = pij
        porfolios_entropy_mc[portfolio] = dimension_country

        stop_mc = time.time()
        print("Total time mc = " + str(float(stop_mc - stop_country)) + " seconds")

        # entropy price-book dimension
        dimension_pb = {}
        for year in years:
            for month in months:
                pij = 0
                for quintil in range(5):
                    key = str(year) + "-" + str(month)
                    stocks = df_pb_quint[key][df_pb_quint[key] == quintil].index
                    len_stocks = len(stocks)
                    for stock in stocks:
                        if (key in volatility_by_month):
                            pij += (volatility_by_month[key][stock] * stock_weights[stock]) / len_stocks
                pij = pij * np.log(np.nan_to_num(pij))
                dimension_pb[key] = pij
        porfolios_entropy_pb[portfolio] = dimension_pb
        stop_pb = time.time()
        print("Total time pb = " + str(float(stop_pb - stop_mc)) + " seconds")

        print("Portfolio {} : DONE in {} seconds".format(portfolio, str(float(time.time() - start_portfolio))))

    print("Total time = " + str(float(time.time() - start)) + " seconds")

    portfolio_entropy = {}
    max_entropy_by_period_value = {}
    max_entropy_by_period_portfolio = {}
    for portfolio in range(num_portfolios):
        portfolio_entropy[portfolio] = {}
        for year in years:
            for month in months:
                key = str(year) + "-" + str(month)
                sum_total = 0
                sum_total += porfolios_entropy_sector[portfolio][key] + porfolios_entropy_country[portfolio][key]
                sum_total += porfolios_entropy_mc[portfolio][key] + porfolios_entropy_pb[portfolio][key]
                sum_total *= -1
                portfolio_entropy[portfolio][key] = sum_total
                if key not in max_entropy_by_period_value.keys():
                    max_entropy_by_period_value[key] = sum_total
                    max_entropy_by_period_portfolio[key] = portfolio
                else:
                    if (sum_total > max_entropy_by_period_value[key] ):
                        # print("In period {} {} of portfolio {} is bigger than {} of portfolio {}".format(
                        #     key,
                        #     sum_total,
                        #     portfolio,
                        #     max_entropy_by_period_value[key],
                        #     max_entropy_by_period_portfolio[key]
                        # ) )
                        max_entropy_by_period_portfolio[key] = portfolio
                    max_entropy_by_period_value[key] = max(sum_total, max_entropy_by_period_value[key])


    print(max_entropy_by_period_portfolio)
    print(max_entropy_by_period_value)
    print("\n")
    final_weights = {}
    for period, portfolio in max_entropy_by_period_portfolio.items():
        final_weights[period] = stock_all_weights[portfolio]
    print(final_weights)

    i=0
    j=60
    returns_all = []
    volatility_all = []
    sharpe_ratio = []
    sortino_list_1 = []
    sortino_list_2 = []
    weights_best = np.array(final_weights)
    # print(weights_best)
    # print(final_weights)

    for period, weights in final_weights:
        weights.values().mean
    #
    # for weights in final_weights.values():
    #     #     print(weights.values())
    #     #     print(type(avg_returns))
    #
    #     avg_returns = returns.iloc[i+60:j+12].mean().to_frame()
    #     #     print(avg_returns)
    #     weights = pd.DataFrame.from_dict(weights.values())
    #     returns_max = returns.iloc[i+60:j+12].mean(axis=1)
    #
    #     weighted_ret = avg_returns.values * weights.values
    #     weighted_ret = weighted_ret.tolist()
    #     weighted_ret = [val for sublist in weighted_ret for val in sublist]
    #
    #     returns_p = (sum(weighted_ret))*12
    #     #     print(returns_p)
    #     returns_all.append(returns_p)
    #
    #
    #     returns_sortino = returns.iloc[i:j]
    #     weights = pd.DataFrame(weights)
    #     weights_list = weights.values.tolist()
    #     weights_list = [val for sublist in weights_list for val in sublist]
    #
    #
    #     weighted_sortino = returns_sortino *  weights_list
    #     #     print(weighted_sortino)
    #     returns_p_sortino = weighted_sortino.sum(axis=1)
    #
    #     returns_p_sortino = pd.DataFrame(returns_p_sortino)
    #     rf_sortino = riskfree.iloc[i:j] / 100
    #
    #
    #     #     print(len(returns_p_sortino.values))
    #     #     print(len(rf_sortino.values))
    #     #     break
    #     excess_returns_sortino = returns_p_sortino.values - rf_sortino.values
    #
    #
    #     cov_matrix = returns.iloc[i:j].cov()
    #     var = np.dot(weights.T,np.dot(cov_matrix,weights))
    #     sd = np.sqrt(var)
    #     vol_p = sd*np.sqrt(12)
    #     volatility_all.append(vol_p)
    #
    #     rf = float(riskfree.iloc[j]) / 100
    #     s_r = (returns_p - rf) / vol_p
    #     s_r = np.round(s_r,3)
    #     sharpe_ratio.append(s_r)
    #
    #
    #
    #     lpm_1 = (np.maximum(0,  rf - np.asarray(excess_returns_sortino)) ** 2).mean(axis=0)
    #     lpm_2 = (np.maximum(0,  0 -np.asarray(excess_returns_sortino)) ** 2).mean(axis=0)
    #
    #     sort1 = ( returns_p - rf) / np.sqrt(lpm_1)
    #     sort1 = np.round(sort1,3).tolist()
    #     sortino_list_1.append(sort1)
    #
    #     sort2 = ( returns_p - rf) / np.sqrt(lpm_2)
    #     sort2 = np.round(sort2,3).tolist()
    #     sortino_list_2.append(sort2)
    #
    #
    #     i+=12
    #     j+=12
    #
    # print(sharpe_ratio)
    # print(sortino_list_1)
    # print(sortino_list_2)
    # print(returns_all)
    # print(volatility_all)

if __name__ == '__main__':
    main_loop(num_portfolios=5)
