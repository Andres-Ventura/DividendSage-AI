import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
import requests

from ..interface import FinancialDataProvider


class AlphaVantageProvider(FinancialDataProvider):
    """Alpha Vantage implementation of FinancialDataProvider."""

    def __init__(self):
        self.router = APIRouter()
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_data(self, symbol: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Generic method to fetch data from Alpha Vantage."""
        try:
            params.update({"apikey": self.api_key, "symbol": symbol})
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self._handle_error(data, symbol)
            return data
        except requests.RequestException:
            raise HTTPException(status_code=503, detail="Failed to fetch data from Alpha Vantage")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_dividend_history(self, symbol: str) -> Dict[str, Any]:
        """Get dividend history for a symbol."""
        data = await self.fetch_data(symbol, {
            "function": "TIME_SERIES_MONTHLY_ADJUSTED",
            "outputsize": "compact"
        })
        monthly_data = data.get("Monthly Adjusted Time Series", {})

        dividend_history = [
            self._format_dividend_data(values, date)
            for date, values in monthly_data.items()
            if self._format_dividend_data(values, date)
        ]

        return {
            "symbol": symbol,
            "dividend_history": sorted(dividend_history, key=lambda x: x["date"], reverse=True)
        }

    async def get_daily_adjusted(self, symbol: str, days: int) -> Dict[str, Any]:
        """Get daily adjusted price and dividend data."""
        data = await self.fetch_data(symbol, {
            "function": "TIME_SERIES_DAILY",
            "outputsize": "compact"
        })
        daily_data = data.get("Time Series (Daily)", {})

        formatted_data = [
            {
                "date": date,
                "adjusted_close": float(values.get("5. adjusted close", 0)),
                "close": float(values.get("4. close", 0)),
                "dividend_amount": float(values.get("7. dividend amount", 0)),
                "split_coefficient": float(values.get("8. split coefficient", 1))
            }
            for date, values in list(daily_data.items())[:days]
        ]

        return {
            "symbol": symbol,
            "daily_data": sorted(formatted_data, key=lambda x: x["date"], reverse=True)
        }

    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Get cash flow statements."""
        data = await self.fetch_data(symbol, {"function": "CASH_FLOW"})
        return {
            "symbol": symbol,
            "quarterly_cash_flow": data.get("quarterlyReports", []),
            "annual_cash_flow": data.get("annualReports", [])
        }

    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings data."""
        data = await self.fetch_data(symbol, {"function": "EARNINGS"})
        return {
            "symbol": symbol,
            "quarterly_earnings": data.get("quarterlyEarnings", []),
            "annual_earnings": data.get("annualEarnings", [])
        }

    async def get_company_stats(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive company statistics."""
        overview_data = await self.fetch_data(symbol, {"function": "OVERVIEW"})
        earnings_data = await self.get_earnings(symbol)

        dividend_per_share = float(overview_data.get("DividendPerShare", 0))
        shares_outstanding = float(overview_data.get("SharesOutstanding", 0))
        eps = float(overview_data.get("EPS", 0))

        ma_50 = float(overview_data.get("50DayMovingAverage", 0))
        ma_200 = float(overview_data.get("200DayMovingAverage", 0))
        current_price = ma_50 if ma_50 > 0 else ma_200

        quarterly_earnings = earnings_data.get("quarterlyEarnings", [])
        earnings_growth = self._calculate_earnings_growth(quarterly_earnings)

        return {
            "symbol": overview_data.get("Symbol"),
            "company_info": self._extract_company_info(overview_data),
            "dividend_metrics": self._calculate_dividend_metrics(
                overview_data, dividend_per_share, shares_outstanding, eps
            ),
            "performance_metrics": self._calculate_performance_metrics(
                overview_data, earnings_growth
            ),
            "technical_indicators": self._calculate_technical_indicators(
                overview_data, dividend_per_share, ma_50, ma_200, current_price
            )
        }

    # Private helper methods
    def _handle_error(self, data: Dict[str, Any], symbol: str):
        """Handle API errors."""
        if "Error Message" in data or not data:
            raise HTTPException(status_code=404, detail=f"Data for {symbol} not found.")
        if "Note" in data:
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    def _format_dividend_data(self, values: Dict[str, str], date: str) -> Optional[Dict[str, Any]]:
        """Format dividend data entry."""
        dividend_amount = float(values.get("7. dividend amount", 0))
        if dividend_amount > 0:
            return {
                "date": date,
                "amount": dividend_amount,
                "adjusted_close": float(values.get("5. adjusted close", 0)),
            }
        return None

    def _calculate_earnings_growth(self, quarterly_earnings: List[Dict[str, str]]) -> float:
        """Calculate earnings growth."""
        
        if len(quarterly_earnings) >= 2:
            current_earnings = float(quarterly_earnings[0].get("reportedEPS", 0))
            prev_earnings = float(quarterly_earnings[1].get("reportedEPS", 0))
            if prev_earnings > 0:
                return ((current_earnings - prev_earnings) / prev_earnings) * 100
        return 0

    def _extract_company_info(self, data: Dict[str, str]) -> Dict[str, str]:
        """Extract company information."""
        
        return {
            "name": data.get("Name"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            "country": data.get("Country"),
            "exchange": data.get("Exchange"),
            "currency": data.get("Currency"),
            "fiscal_year_end": data.get("FiscalYearEnd"),
        }

    def _calculate_dividend_metrics(
        self, data: Dict[str, str], dividend_per_share: float, shares_outstanding: float, eps: float
    ) -> Dict[str, Any]:
        """Calculate dividend-related metrics."""
        
        return {
            "dividend_yield": float(data.get("DividendYield", 0)) * 100,
            "dividend_per_share": dividend_per_share,
            "annual_dividend_amount": dividend_per_share * shares_outstanding if shares_outstanding > 0 else 0,
            "payout_ratio": (dividend_per_share / eps * 100) if eps > 0 else 0,
            "next_dividend_date": data.get("DividendDate"),
            "ex_dividend_date": data.get("ExDividendDate"),
        }

    def _calculate_performance_metrics(self, data: Dict[str, str], earnings_growth: float) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        return {
            "market_cap": float(data.get("MarketCapitalization", 0)),
            "pe_ratio": float(data.get("PERatio", 0)),
            "forward_pe": float(data.get("ForwardPE", 0)),
            "peg_ratio": float(data.get("PEGRatio", 0)),
            "profit_margin": float(data.get("ProfitMargin", 0)) * 100,
            "operating_margin": float(data.get("OperatingMarginTTM", 0)) * 100,
            "return_on_equity": float(data.get("ReturnOnEquityTTM", 0)) * 100,
            "quarterly_earnings_growth": earnings_growth,
        }

    def _calculate_technical_indicators(
        self, data: Dict[str, str], dividend_per_share: float, ma_50: float, ma_200: float, current_price: float
    ) -> Dict[str, Any]:
        """Calculate technical indicators."""
        return {
            "52_week_high": float(data.get("52WeekHigh", 0)),
            "52_week_low": float(data.get("52WeekLow", 0)),
            "moving_average_50day": ma_50,
            "moving_average_200day": ma_200,
            "price_to_dividend_ratio": current_price / dividend_per_share if dividend_per_share > 0 else 0,
            "distance_from_50ma": ((current_price / ma_50 - 1) * 100) if ma_50 > 0 else 0,
            "distance_from_200ma": ((current_price / ma_200 - 1) * 100) if ma_200 > 0 else 0,
        }