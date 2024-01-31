import yaml
import yfinance as yf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import re
from collections import defaultdict

CONFIG_FNAME = 'config.yaml'

def load_config():
    # Load configuration from YAML file
    config_dict = {}
    with open(CONFIG_FNAME, 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_dict['tickers'] = config['portfolio_optimisation']['tickers']
        config_dict['years_history'] = config['portfolio_optimisation']['years_history']
        config_dict['risk_free_rate'] = config['portfolio_optimisation']['risk_free_rate']
        config_dict['num_portfolios'] = config['portfolio_optimisation']['num_portfolios']
        config_dict['base_value'] = config['portfolio_optimisation']['base_value']
        config_dict['include_dividends_in_returns'] = config['portfolio_optimisation']['include_dividends_in_returns']
        config_dict['start_date'] = config['portfolio_optimisation']['start_date']
        config_dict['use_prev_year'] = config['portfolio_optimisation']['use_prev_year']
        
        config_dict['host'] = config['app']['host']
        config_dict['port'] = config['app']['port']
        config_dict['debug'] = config['app']['debug']

    return config_dict

def get_stock_data(tickers,
                   start_date,
                   end_date):
    """
    Get stock data from Yahoo Finance.
    """

    # Initialize dictionaries to store data
    closing_prices = {}
    dividends = {}

    # Fetch data
    for ticker in tickers:
        stock_data = yf.Ticker(ticker)
        hist_data = stock_data.history(start=start_date, end=end_date)
        
        # Store closing prices and dividends
        closing_prices[ticker + '_price'] = hist_data['Close']
        dividends[ticker + '_div'] = hist_data['Dividends']

    # Create DataFrames
    closing_prices_df = pd.DataFrame(closing_prices)
    dividends_df = pd.DataFrame(dividends)

    # Combine the DataFrames
    raw_data = pd.concat([closing_prices_df, dividends_df], axis=1)

    return raw_data

def process_data(df, tickers):
    """
    Process and clean the raw data.
    """
    
    price_cols = [x + '_price' for x in tickers] 
    divy_cols = [x + '_divyield' for x in tickers]

    # Calculate divided yields
    df_copy = df.copy()
    for x in tickers:
        df_copy[x + '_divyield'] = df_copy[x + '_div'] / df_copy[x + '_price']

    # Fill NAs for dividend yields
    df_copy[divy_cols] = df_copy[divy_cols].fillna(0)

    # Calculate daily returns and drop NAs
    df_copy[price_cols] = df_copy[price_cols].pct_change()
    df_copy.dropna(inplace=True)

    # Create output dataframes
    daily_returns = df_copy[price_cols]
    daily_dividend_yields = df_copy[divy_cols]

    daily_returns.columns = tickers
    daily_dividend_yields.columns = tickers

    daily_total_returns = daily_returns + daily_dividend_yields
    
    return daily_returns, daily_dividend_yields, daily_total_returns

def normalise_data(raw_data):
    normalised_data = raw_data.copy()
    normalised_data.dropna(inplace=True)
    normalised_data = normalised_data / normalised_data.iloc[0]

    return normalised_data

def generate_portfolios(daily_returns, 
                        daily_dividend_yields, 
                        risk_free_rate=0.0,
                        num_portfolios=25000,
                        include_dividends_in_returns=True
                        ):
    """"
    Generate Portfolios.
    """
    # Annualize the daily returns and dividend yields
    mean_daily_returns = daily_returns.mean()
    if include_dividends_in_returns:
        mean_daily_dividend_yields = daily_dividend_yields.mean()
    else:
        mean_daily_dividend_yields = 0.0
    annual_returns = (1 + mean_daily_returns + mean_daily_dividend_yields).pow(252) - 1

    cov_matrix = daily_returns.cov()

    # Set the number of iterations for the simulation

    num_stocks = len(daily_returns.columns)
    results = np.zeros((3 + num_stocks, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        
        # Adjusted for dividends
        portfolio_return = np.sum(annual_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_stddev  # Sharpe Ratio
        
        for j in range(len(weights)):
            results[j+3,i] = weights[j]

    columns = ['ret', 'stdev', 'sharpe'] + [ticker + '_weight' for ticker in daily_returns.columns]
    results_df = pd.DataFrame(results.T, columns=columns)

    return results_df

def get_optimal_portfolios(results_df):
    """
    Find the portfolio with the highest Sharpe Ratio and minimum standard deviation
    """
    max_sharpe_port = results_df.iloc[results_df['sharpe'].idxmax()]
    min_vol_port = results_df.iloc[results_df['stdev'].idxmin()]

    return max_sharpe_port, min_vol_port

def create_combined_port_df(max_sharpe_port, min_vol_port):
    """
    Create combined portfolio dataframe.
    """
    max_sharpe_port_data = max_sharpe_port.to_frame().T.round(2).reset_index(drop=True)
    max_sharpe_port_data.insert(0, 'portfolio', 'max sharpe')
    column_names = [re.split(r'[._]', x)[0] for x in max_sharpe_port_data.columns.to_list()]
    max_sharpe_port_data.columns = column_names

    min_vol_port_data = min_vol_port.to_frame().T.round(2).reset_index(drop=True)
    min_vol_port_data.insert(0, 'portfolio', 'min volatility')
    min_vol_port_data.columns = column_names

    combined_data = pd.concat([max_sharpe_port_data, min_vol_port_data])
    combined_data = combined_data.to_dict(orient='records')

    return combined_data

def backtest_portfolio_optimisation(tickers,
                                    daily_returns, 
                                    daily_dividend_yields,
                                    risk_free_rate,
                                    num_portfolios,
                                    include_dividends_in_returns,
                                    use_prev_year=True):
    """
    Backtest portfolio optimisation.
    """

    # Lists to store the optimal portfolio returns after rebalancing
    max_sharpe_returns = []
    min_vol_returns = []
    max_sharpe_weights_dict = defaultdict(list)

    # Init dates
    start_date = min(daily_returns.index).strftime('%Y-%m-%d')
    year_next = int(min(daily_returns.index).strftime('%Y')) + 1
    year_max = int(max(daily_returns.index).strftime('%Y'))
    first_start_date = None

    counter = 0
    while year_next <= year_max + 1:
        if counter == 1:
            first_start_date = start_date
        end_date = f'{year_next}-03-31'
        print(start_date, end_date)

        # Select rows between two dates
        daily_returns_ = daily_returns[(daily_returns.index>= start_date) & (daily_returns.index <= end_date)]
        daily_dividend_yields_ = daily_dividend_yields[(daily_dividend_yields.index>= start_date) & (daily_dividend_yields.index <= end_date)]

        if daily_returns_.shape[0] > 0:
            if use_prev_year:
                results_df = generate_portfolios(daily_returns_, 
                                                daily_dividend_yields_, 
                                                risk_free_rate=risk_free_rate,
                                                num_portfolios=num_portfolios,
                                                include_dividends_in_returns=include_dividends_in_returns
                                                )
            else:
                daily_returns_cum = daily_returns[daily_returns.index <= end_date]
                daily_dividend_yields_cum = daily_dividend_yields[daily_dividend_yields.index <= end_date]
                results_df = generate_portfolios(daily_returns_cum, 
                                    daily_dividend_yields_cum, 
                                    risk_free_rate=risk_free_rate,
                                    num_portfolios=num_portfolios,
                                    include_dividends_in_returns=include_dividends_in_returns
                                    )    
                
            max_sharpe_port, min_vol_port = get_optimal_portfolios(results_df)

            if counter > 0:
                max_sharpe_weighted_daily_returns = list(np.sum(daily_returns_.values * max_sharpe_weights, axis=1))
                min_vol_weighted_daily_returns = list(np.sum(daily_returns_.values * min_vol_weights, axis=1))
                max_sharpe_returns.extend(max_sharpe_weighted_daily_returns)
                min_vol_returns.extend(min_vol_weighted_daily_returns)

                max_sharpe_weights_dict['date'].append(start_date)
                for x in tickers:
                    max_sharpe_weights_dict[re.split(r'[._]', x)[0]].append(max_sharpe_port[x + '_weight'])

            # Update weights
            max_sharpe_weights = max_sharpe_port[[x + '_weight' for x in tickers]].values
            min_vol_weights = min_vol_port[[x + '_weight' for x in tickers]].values

        # Update dates
        counter += 1
        start_date = f'{year_next}-04-01'
        year_next += 1

    # Create dataframe of weights as portfolio is rebalanced over time
    max_sharpe_weights_data = pd.DataFrame(max_sharpe_weights_dict).round(2).to_dict(orient='records')

    return max_sharpe_returns, min_vol_returns, max_sharpe_weights_data, first_start_date

def create_price_charts(raw_data, tickers):
    """
    Create plotly price charts.
    """
    dates = raw_data.index
    fig = go.Figure()
    for ticker in tickers:
        prices = raw_data[ticker + '_price']
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name=ticker))
    fig.update_layout(width=1000, 
                    height=600,     
                    paper_bgcolor='#D5DEED',
                    font={'color':'#0B1C48'},
                    title={'text':'Normalised Price History', 
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
                    xaxis_title='Date', 
                    yaxis_title='Cumulative Returns',
                    legend ={'title_text': '',
                        'bgcolor': 'rgba(0,0,0,0)',
                        'orientation': 'h',
                        'yanchor': 'bottom',
                        'y': -0.3,
                        'xanchor': 'center',
                        'x': 0.5})
    return fig

def create_efficient_frontier_chart(results_df, 
                                    max_sharpe_port,
                                    min_vol_port):
    # Scatter plot of the efficient frontier
    fig = px.scatter(results_df, x='stdev', y='ret', color='sharpe',
                    color_continuous_scale='Inferno'
                    )

    # Add marker for the portfolio with the highest Sharpe Ratio
    fig.add_trace(go.Scatter(x=[max_sharpe_port['stdev']], y=[max_sharpe_port['ret']],
                            mode='markers+text', text=["Max Sharpe Ratio"], textposition="top center",
                            marker=dict(color='red', size=20, line=dict(width=2, color='DarkSlateGrey')),
                            name='Max Sharpe Ratio'))

    # Add marker for the portfolio with minimum volatility
    fig.add_trace(go.Scatter(x=[min_vol_port['stdev']], y=[min_vol_port['ret']],
                            mode='markers+text', text=["Min Volatility"], textposition="bottom center",
                            marker=dict(color='green', size=20, line=dict(width=2, color='DarkSlateGrey')),
                            name='Min Volatility'))

    # Update layout for labels and title
    fig.update_layout(width=1000, 
                      height=600, 
                      paper_bgcolor='#D5DEED',
                      font={'color':'#0B1C48'},
                      xaxis_title='Volatility', 
                      yaxis_title='Return',
                      coloraxis_colorbar=dict(title='Sharpe Ratio'),
                      showlegend=False,
                      title={'text':'Efficient Frontier with Optimal Portfolios', 
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}
                    )

    return fig

def create_backtest_chart(daily_returns, 
                          first_start_date, 
                          max_sharpe_returns,
                          min_vol_returns,
                          base_value=100):

    daily_returns_ = daily_returns[(daily_returns.index>=first_start_date)]
    max_sharpe_returns_np = np.array(max_sharpe_returns)
    min_vol_returns_np = np.array(min_vol_returns)
    max_sharpe_multiplicative_returns = 1.0 + max_sharpe_returns_np
    min_vol_multiplicative_returns = 1.0 + min_vol_returns_np
    max_sharpe_cumulative_returns = np.cumprod(max_sharpe_multiplicative_returns)
    min_vol_cumulative_returns = np.cumprod(min_vol_multiplicative_returns)
    # Apply cumulative returns to base value
    max_sharpe_final_values = base_value * max_sharpe_cumulative_returns
    min_vol_final_values = base_value * min_vol_cumulative_returns

    dates = daily_returns_.index
    
    # Plotting each asset
    fig = go.Figure()
    for col in daily_returns_.columns:
        multiplicative_returns = 1 + daily_returns_[col].values
        cumulative_returns = np.cumprod(multiplicative_returns)
        final_values = base_value * cumulative_returns
        fig.add_trace(go.Scatter(x=dates, y=final_values, mode='lines', name=col, opacity=1.0))

    # Plot optimal portfolios
    fig.add_trace(go.Scatter(x=dates, y=max_sharpe_final_values, mode='lines', name='Max Sharpe', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=min_vol_final_values, mode='lines', name='Min Volatility', line=dict(dash='dash')))

    for year in range(dates.year.min(),  dates.year.max()+1):
        date_str = f"{year}-04-01"
        fig.add_vline(x=date_str, line_width=1, line_dash="dash", line_color="darkgrey")

    fig.update_layout(width=1000, 
                      height=600, 
                      paper_bgcolor='#D5DEED',
                      font={'color':'#0B1C48'},
                      title={'text':'Cumulative Returns of Optimal Portfolios and Assets', 
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}, 
                      xaxis_title='Date', 
                      yaxis_title='Cumulative Returns',
                      legend ={'title_text': '',
                            'bgcolor': 'rgba(0,0,0,0)',
                            'orientation': 'h',
                            'yanchor': 'bottom',
                            'y': -0.3,
                            'xanchor': 'center',
                            'x': 0.5})
    
    return fig

# def update_data():
#     raw_data = ma.get_api_data()
#     formatted_data = ut.process_mews_data(raw_data)
#     db.insert_into_reservations_table(formatted_data)