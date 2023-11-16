from flask import Flask, render_template, request
import pandas as pd
import plotly
import json
import utils as ut
import re

app = Flask(__name__)

def get_data_for_template(tickers, 
                                 start_date,
                                 end_date,
                                 risk_free_rate,
                                 num_portfolios,
                                 base_value,
                                 include_dividends_in_returns,
                                 use_prev_year):
    
    raw_data = ut.get_stock_data(tickers,
                                start_date,
                                end_date)
    
    daily_returns, \
    daily_dividend_yields, \
    daily_total_returns = ut.process_data(raw_data, tickers)

    normalised_data = ut.normalise_data(raw_data)

    results_df = ut.generate_portfolios(daily_returns, 
                                        daily_dividend_yields, 
                                        risk_free_rate=risk_free_rate,
                                        num_portfolios=num_portfolios,
                                        include_dividends_in_returns=include_dividends_in_returns)
    
    max_sharpe_port, min_vol_port = ut.get_optimal_portfolios(results_df)
    combined_data = ut.create_combined_port_df(max_sharpe_port, min_vol_port)
    
    # Backtest strategy
    max_sharpe_returns, \
    min_vol_returns, \
    max_sharpe_weights_data, \
    first_start_date = ut.backtest_portfolio_optimisation(tickers,
                                                            daily_returns, 
                                                            daily_dividend_yields,
                                                            risk_free_rate,
                                                            num_portfolios,
                                                            include_dividends_in_returns,
                                                            use_prev_year=use_prev_year)
    
    # Process the data and create Plotly figures
    prices_fig = ut.create_price_charts(normalised_data, tickers)
    efficient_frontier_fig = ut.create_efficient_frontier_chart(results_df, 
                                                                max_sharpe_port,
                                                                min_vol_port)

    backtest_fig = ut.create_backtest_chart(daily_returns, 
                                            first_start_date, 
                                            max_sharpe_returns,
                                            min_vol_returns,
                                            base_value=base_value)
    
    # Convert the figures to JSON for rendering in the HTML template
    prices_fig_json = json.dumps(prices_fig, cls=plotly.utils.PlotlyJSONEncoder)
    efficient_frontier_fig_json = json.dumps(efficient_frontier_fig, cls=plotly.utils.PlotlyJSONEncoder)
    backtest_fig_json = json.dumps(backtest_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return prices_fig_json, efficient_frontier_fig_json, backtest_fig_json, combined_data, max_sharpe_weights_data

# Route for serving the dashboard
# http://127.0.0.1:5000/
@app.route('/', methods=['GET', 'POST'])
def dashboard():

    if request.method == 'POST':
        text_data = request.form['textfield']
        tickers = re.sub(r'[^a-zA-Z. ]', '', text_data)
        tickers = re.sub(r'\s+', ' ', tickers).split(' ')[:10]
        if request.form['dropdown2'] == 'div_incl':
            include_dividends_in_returns = True
        else:
            include_dividends_in_returns = False
        if request.form['dropdown3'] == 'rw_prev_year':
            use_prev_year = True
        else:
            use_prev_year = False

        text_start_date = request.form['textstartdate']
        dropdown2 = request.form['dropdown2']
        dropdown3 = request.form['dropdown3']

        # Load config dict
        config = ut.load_config()
        risk_free_rate = config['risk_free_rate']
        num_portfolios = config['num_portfolios']
        base_value = config['base_value']
        include_dividends_in_returns = config['include_dividends_in_returns']

        # Ensure text_start_date is valid, otherwise set to config default
        try:
            pd.Timestamp(text_start_date)
        except ValueError:
            text_start_date = config['start_date']

    else:
        # Load config dict
        config = ut.load_config()
        tickers = config['tickers']
        risk_free_rate = config['risk_free_rate']
        num_portfolios = config['num_portfolios']
        base_value = config['base_value']
        include_dividends_in_returns = config['include_dividends_in_returns']
        text_start_date = config['start_date']
        use_prev_year = config['use_prev_year']
        text_data = ' '.join(tickers)

        dropdown2 = 'div_incl'
        dropdown3 = 'rw_prev_year'

    # Time period
    start_date = pd.Timestamp(text_start_date)
    end_date = pd.Timestamp.now()

    # Get figures
    prices_fig_json, \
    efficient_frontier_fig_json, \
    backtest_fig_json, \
    combined_data, \
    max_sharpe_weights_data = get_data_for_template(tickers, 
                                start_date,
                                end_date,
                                risk_free_rate,
                                num_portfolios,
                                base_value,
                                include_dividends_in_returns,
                                use_prev_year)
       
    # Render the template, passing the data and JSON strings
    return render_template('portfolio_optimisation.html',
                            prices_fig_json=prices_fig_json,
                            efficient_frontier_fig_json=efficient_frontier_fig_json,
                            backtest_fig_json=backtest_fig_json,
                            combined_data=combined_data,
                            max_sharpe_weights_data=max_sharpe_weights_data,
                            text_data=text_data,
                            dropdown2=dropdown2,
                            dropdown3=dropdown3,
                            text_start_date=text_start_date)

if __name__ == '__main__':
    # Load config dict
    config = ut.load_config()
    app.run(host=config['host'], port=config['port'], debug=config['debug'])
