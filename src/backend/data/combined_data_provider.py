from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from backend.data.extract.alpha_vantage import AlphaVantageProvider
from backend.data.extract.pdf_scanner import FileUploadParser
from backend.data.extract.yfinance import YFinanceAPI


class CombinedDataProvider:
    """
    Aggregates and preprocesses data from Alpha Vantage, file uploads, and Yahoo Finance.
    """

    def __init__(self):
        self.alpha_provider = AlphaVantageProvider()
        self.upload_parser = FileUploadParser()
        self.yfinance_provider = None  # Initialized dynamically per symbol

    async def fetch_combined_data(self, symbol: str, uploaded_file: Optional[str] = None) -> dict[str, Any]:
        """
        Fetch and aggregate data from multiple sources.

        Args:
            symbol (str): Stock symbol to fetch data for.
            uploaded_file (Optional[str]): Path to an uploaded file for parsing.

        Returns:
            Dict[str, Any]: Aggregated data from all sources.
        """
        try:
            # Alpha Vantage data
            alpha_dividends = await self.alpha_provider.get_dividend_history(symbol)
            alpha_daily = await self.alpha_provider.get_daily_adjusted(symbol, days=30)

            # Yahoo Finance data
            self.yfinance_provider = YFinanceAPI(symbol)
            yf_historical = self.yfinance_provider.get_historical_data_yf(period="1y", interval="1d")
            yf_dividends = self.yfinance_provider.get_dividends_and_splits_yf()
            yf_earnings = self.yfinance_provider.get_corporate_actions_yf()

            # Uploaded file data
            uploaded_data = None
            if uploaded_file:
                uploaded_data = self.upload_parser.parse_file(uploaded_file)

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

            return combined_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch combined data: {str(e)}")
