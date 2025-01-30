from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from backend.data.extract.alpha_vantage import AlphaVantageProvider
from backend.data.extract.pdf_scanner import FileUploadParser
from backend.data.extract.yfinance import YFinanceAPI
import logging


class CombinedDataProvider:
    """
    Aggregates and preprocesses data from Alpha Vantage, file uploads, and Yahoo Finance.
    """

    def __init__(self):
        self.alpha_provider = AlphaVantageProvider()
        self.upload_parser = FileUploadParser()
        self.yfinance_provider = None  # Initialized dynamically per symbol

    async def fetch_combined_data(
        self, symbol: str, uploaded_file: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Fetch and aggregate data from multiple sources.

        Args:
            symbol (str): Stock symbol to fetch data for.
            uploaded_file (Optional[str]): Path to an uploaded file for parsing.

        Returns:
            Dict[str, Any]: Aggregated data from all sources.
        """
        logging.info(f"Starting data fetch for symbol: {symbol}")
        try:
            # Alpha Vantage data
            logging.info("Fetching Alpha Vantage dividend history...")
            alpha_dividends = await self.alpha_provider.get_dividend_history(symbol)
            logging.info("Fetching Alpha Vantage daily adjusted data...")
            alpha_daily = await self.alpha_provider.get_daily_adjusted(symbol, days=30)
            logging.debug(
                f"Alpha Vantage data received - Dividends: {bool(alpha_dividends)}, Daily: {bool(alpha_daily)}"
            )

            # Yahoo Finance data
            logging.info("Initializing Yahoo Finance provider...")
            self.yfinance_provider = YFinanceAPI(symbol)
            logging.info("Fetching Yahoo Finance historical data...")
            yf_historical = self.yfinance_provider.get_historical_data_yf("max", "1d")

            logging.info("Fetching Yahoo Finance dividends and splits...")
            yf_dividends = self.yfinance_provider.get_dividends_and_splits_yf()
            logging.info("Fetching Yahoo Finance earnings data...")
            yf_earnings = self.yfinance_provider.get_corporate_actions_yf()
            logging.debug(
                f"Yahoo Finance data received - Historical: {not yf_historical.empty}, Dividends: {bool(yf_dividends)}, Earnings: {bool(yf_earnings)}"
            )

            # Uploaded file data
            uploaded_data = None
            if uploaded_file:
                logging.info(f"Parsing uploaded file: {uploaded_file}")
                uploaded_data = self.upload_parser.parse_file(uploaded_file)
                logging.debug(f"Upload data parsed: {bool(uploaded_data)}")

            # Combine data into a unified format
            combined_data = {
                "symbol": symbol,
                "alpha_vantage": {
                    "dividends": alpha_dividends,
                    "daily_adjusted": alpha_daily,
                },
                "yahoo_finance": {
                    "historical": yf_historical.to_dict(),
                    "dividends_and_splits": yf_dividends,
                    "earnings": yf_earnings,
                },
                "uploaded_data": uploaded_data,
            }

            logging.info("Data combination complete")
            logging.debug(f"Combined data structure: {combined_data.keys()}")
            return combined_data
        except Exception as e:
            logging.error(f"Error in fetch_combined_data: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch combined data: {str(e)}"
            )
