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
        
        ticker = yf.Ticker(self.symbol)
        historical_data = ticker.history(period=period, interval=interval)
        
        return historical_data
    
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
        """
        Get information on annual and quarterly earnings
        """
        
        ticker = yf.Ticker(self.symbol)
        earnings = ticker.earnings
        quarterly_earnings = ticker.quarterly_earnings
        financials = ticker.financials
        
        return {
            "annual_earnings": earnings, 
            "quarterly_earnings": quarterly_earnings,
            "financials": financials
        }
    
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