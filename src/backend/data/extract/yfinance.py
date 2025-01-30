import logging
from typing import Dict, Any, Optional
import pandas as pd
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
    
    def get_corporate_actions_yf(self) -> dict:
        """
    Fetch corporate actions including dividends, stock splits, and earnings data from Yahoo Finance.

    Returns:
        dict: Processed corporate actions data.
    """
        try:
            ticker = yf.Ticker(self.symbol)
        
            # Process dividends and stock splits
            dividends = ticker.dividends.reset_index()
            dividends.columns = ["date", "amount"]
            dividends_dict = dividends.to_dict(orient="records")

            splits = ticker.splits.reset_index()
            splits.columns = ["date", "split_ratio"]
            splits_dict = splits.to_dict(orient="records")

            # Initialize earnings data structures
            quarterly_earnings = pd.DataFrame()
            annual_earnings = []

            # Fetch and process quarterly earnings
            try:
                quarterly_earnings = ticker.earnings_dates
                if not quarterly_earnings.empty:
                    # Dynamically handle column names to avoid KeyError
                    rename_mapping = {}
                    if "Earnings Date" in quarterly_earnings.columns:
                        rename_mapping["Earnings Date"] = "date"
                    if "EPS Estimate" in quarterly_earnings.columns:
                        rename_mapping["EPS Estimate"] = "eps_estimate"
                    if "Reported EPS" in quarterly_earnings.columns:
                        rename_mapping["Reported EPS"] = "reported_eps"
                    if "Surprise(%)" in quarterly_earnings.columns:
                        rename_mapping["Surprise(%)"] = "surprise_pct"
                
                    quarterly_earnings = quarterly_earnings.rename(columns=rename_mapping)
                    quarterly_earnings = quarterly_earnings.reset_index()
                    quarterly_earnings["date"] = quarterly_earnings["date"].dt.strftime("%Y-%m-%d")
            except KeyError as ke:
                logging.error(f"KeyError encountered while processing earnings data: {ke}. Data may be incomplete.")
            except Exception as e:
                logging.error(f"Error fetching quarterly earnings data: {e}")

            # Fetch and process annual earnings (example, adjust as needed)
            try:
                annual_earnings_df = ticker.earnings
                if not annual_earnings_df.empty:
                    annual_earnings_df = annual_earnings_df.reset_index()
                    annual_earnings_df["Year"] = annual_earnings_df["Year"].astype(str)
                    annual_earnings = annual_earnings_df.to_dict(orient="records")
            except Exception as e:
                logging.error(f"Error fetching annual earnings data: {e}")

            return {
                "dividends": dividends_dict,
                "splits": splits_dict,
                "earnings": {
                    "quarterly": quarterly_earnings.to_dict(orient="records") if not quarterly_earnings.empty else [],
                    "annual": annual_earnings
                }
            }
        except Exception as e:
            logging.error(f"Unexpected error in get_corporate_actions_yf: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching corporate actions data: {str(e)}"
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