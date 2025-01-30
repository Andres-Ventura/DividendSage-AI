from typing import Dict, Any, Optional
import yfinance as yf
from fastapi import HTTPException

from backend.data.interface import FinancialDataProvider

class YFinanceAPI():
    """Yahoo Finance API handler"""
    
    def __init__(self, symbol: str):
        """
        Initialize the API handler for a specific ticker symbol.
        """
        
        self.symbol = symbol
    
    def get_historical_data_yf(self, period: str, interval: str):
        """
        Get historical data for the initialized ticker symbol.
        
        Args:
            period (str): The period for which to retrieve data (e.g., "1d", "5d", "1mo", "1y", "max").
            interval (str): The interval for data points (e.g., "1m", "5m", "1h", "1d").
        
        Returns:
            pandas.DataFrame: Historical market data for the specified period and interval.
        """
        
        try:
            ticker = yf.Ticker(self.symbol)
            historical_data = ticker.history(period=period, interval=interval)
            if historical_data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for symbol {self.symbol}")
            return historical_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    def get_dividends_and_splits_yf(self):
        """
        Gets information on the dividend and stock splits
        """
        
        ticker = yf.Ticker(self.symbol)
        dividends = ticker.dividends
        splits = ticker.splits
        
        return {
            "dividends": dividends, 
            "splits": splits
        }
    
    def get_corporate_actions_yf(self):
        try:
            ticker = yf.Ticker(self.symbol)
        
            # Fetch annual earnings (use income statement for "Net Income")
            income_stmt = ticker.income_stmt
            yearly_earnings = income_stmt.loc["Net Income"] if "Net Income" in income_stmt.index else None
        
            # Fetch quarterly earnings dates and EPS (returns DataFrame with EPS data)
            quarterly_earnings = ticker.earnings_dates
        
            # Fetch financials (annual by default; pass False for quarterly)
            financials_annual = ticker.get_income_stmt()
            financials_quarterly = ticker.get_income_stmt(False)
        
            # Handle NaN values
            if yearly_earnings is not None:
                yearly_earnings = yearly_earnings.fillna(0)
                quarterly_earnings = quarterly_earnings.fillna(0)
                financials_annual = financials_annual.fillna(0)
                financials_quarterly = financials_quarterly.fillna(0)
        
            if yearly_earnings is None or quarterly_earnings.empty:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No corporate actions data found for {self.symbol}"
                )
                
            # Log the return values for debugging with better formatting        
            return {
                "yearly_earnings": yearly_earnings,
                "quarterly_earnings": quarterly_earnings,
                "financials_annual": financials_annual,
                "financials_quarterly": financials_quarterly
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error fetching data: {str(e)}"
            )
    
    def get_analyst_recommendations_yf(self):
        """
        Detailed report of analyst recommendations on a company
        """
        
        ticker = yf.Ticker(self.symbol)
        recommendations = ticker.recommendations
        
        return recommendations
    
    def get_flexible_data_yf(self, asset_type="stock", period="1y"):
        """
        Get data of stock performance in a set period
        """
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period)
        
        return data
    
    def get_news_yf(self):
        """
        Get recent news about company
        """
        
        ticker = yf.Ticker(self.symbol)
        news = ticker.news
        
        return news
    
    def fetch_data_simple_yf(self, period="1mo", interval="1d"):
        """
        Get history of company stock price in set period and interval
        """
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period, interval=interval)
        
        return data