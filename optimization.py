import pandas as pd
import scipy.optimize as spo
import matplotlib.pyplot as plt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data

def assess_portfolio(allocs, prices):
    sv = 1000000.0
    rfr = 0.0
    sf = 252.0
    norm_prices = prices * allocs * sv
    port_val = norm_prices.sum(axis=1)

    period_end = port_val.iloc[-1]
    commul = port_val.pct_change()
    cr = (period_end - sv) / sv
    adr = commul[1:].mean()
    sddr = commul[1:].std()
    sr = np.sqrt(sf) * (commul[1:] - rfr).mean() / ((commul[1:] - rfr).std())
    return [cr, adr, sddr, sr, port_val]

def run_assess_portfolio(allocs,prices):
    return -1.0*assess_portfolio(allocs,prices)[3]

constraints = ({'type': 'ineq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
bounds = ((0.0, 1,0),)


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all['SPY']
    prices = prices / prices.iloc[0, :]

    noa = len(syms)
    allocs = noa * [1.0 / noa, ]
    bnds = tuple((0, 1) for x in range(noa))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    results = spo.minimize(run_assess_portfolio, allocs, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)
    cr, adr, sddr, sr, port_val = assess_portfolio(prices, results.x)
    allocs = results.x
# Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.iloc[0, :]
        #ax = df_temp.plot(title='Daily Portfolio Value and SPY')
        #ax.set_ylabel('Normalized Prices')
        #ax.set_xlabel('Dates')
        #plt.savefig('comparison_optimal.png')
        plot_data(df_temp)

    return allocs, cr, adr, sddr, sr

def get_values(portvals):
        k = np.sqrt(252)

  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
def test_code():  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # This function WILL NOT be called by the auto grader  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Do not assume that any variables defined here are available to your function/code  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # It is only here to help you set up and test your code  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Define input parameters  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Note that ALL of these values will be set to different values by  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # the autograder!  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    start_date = dt.datetime(2021,1,1)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    end_date = dt.datetime(2023,1,1)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    symbols = ['GOOG', 'AAPL', 'MSFT', 'KO', 'TSLA']  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Assess the portfolio  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Print statistics  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Start Date: {start_date}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"End Date: {end_date}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Symbols: {symbols}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Allocations:{allocations}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio: {sr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return: {adr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return: {cr}")  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # This code WILL NOT be called by the auto grader  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Do not assume that it will be called  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  
