from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import requests
import yfinance as yf

router = APIRouter(
    prefix="/api/dividends",
    tags=["dividends"]
) 

@router.get("/{symbol}")
async def get_company_dividends(symbol: str) -> Dict[Any, Any]:
    """
    Get dividend history for a specific company using its stock symbol
    Uses Alpha Vantage's TIME_SERIES_MONTHLY_ADJUSTED endpoint which
    includes dividend data
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey=Y1W0Q5VHJQIQBPT6"

    try:
        response = requests.get(url)
        data = response.json()
        apple = yf.Ticker("aapl").tolist()

        print(f"apple: {apple}")

        if "Error Message" in data:
            raise HTTPException(status_code=404, detail=f"No dividend data found for symbol {symbol}")
        
        # Extract monthly data
        monthly_data = data.get("Monthly Adjusted Time Series", {})

        # Format dividend data
        dividend_history = []
        for date, values in monthly_data.items():
            dividend_amount = float(values.get("4. close", 0))
            if dividend_amount > 0:
                dividend_history.append({
                    "date": date,
                    "amount": dividend_amount
                })

        return {
            "symbol": symbol,
            "dividend_history": dividend_history,
            "yfinance_data": {
                "dividends": apple.dividends
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/{symbol}/stats")
async def get_dividend_stats(symbol: str) -> Dict[Any, Any]:
    """
    Get dividend statistics for a company including yield, frequency, and growth
    Uses Alpha Vantage's OVERVIEW endpoint which includes dividend-related metrics
    """
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey=Y1W0Q5VHJQIQBPT6"

    try:
        response = requests.get(url)
        data = response.json()

        if "Error Message" in data:
            raise HTTPException(status_code=404, detail=f"No dividend stats found for symbol {symbol}")

        dividend_per_share = float(data.get("DividendPerShare", 0))
        current_price = (float(data.get("50DayMovingAverage", 0)) + float(data.get("200DayMovingAverage", 0))) / 2

        return {
            "symbol": data.get("Symbol"),
            "company_info": {
                "name": data.get("Name"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "country": data.get("Country"),
                "exchange": data.get("Exchange"),
                "currency": data.get("Currency")
            },
            "dividend_metrics": {
                "dividend_yield": float(data.get("DividendYield", 0)) * 100,
                "dividend_per_share": dividend_per_share,
                "annual_dividend_amount": dividend_per_share * float(data.get("SharesOutstanding", 0)),
                "payout_ratio": (dividend_per_share / float(data.get("EPS", 1))) * 100,
                "next_dividend_date": data.get("DividendDate"),
                "ex_dividend_date": data.get("ExDividendDate")
            },
            "performance_metrics": {
                "market_cap": float(data.get("MarketCapitalization", 0)),
                "pe_ratio": float(data.get("PERatio", 0)),
                "forward_pe": float(data.get("ForwardPE", 0)),
                "peg_ratio": float(data.get("PEGRatio", 0)),
                "profit_margin": float(data.get("ProfitMargin", 0)) * 100,
            "operating_margin": float(data.get("OperatingMarginTTM", 0)) * 100,
                "return_on_equity": float(data.get("ReturnOnEquityTTM", 0)) * 100,
                "revenue_growth": float(data.get("QuarterlyRevenueGrowthYOY", 0)) * 100,
                "earnings_growth": float(data.get("QuarterlyEarningsGrowthYOY", 0)) * 100
            },
            "risk_metrics": {
                "beta": float(data.get("Beta", 0)),
                "price_to_book": float(data.get("PriceToBookRatio", 0)),
                "price_to_sales": float(data.get("PriceToSalesRatioTTM", 0))
            },
            "technical_indicators": {
                "52_week_high": float(data.get("52WeekHigh", 0)),
                "52_week_low": float(data.get("52WeekLow", 0)),
                "moving_average_50day": float(data.get("50DayMovingAverage", 0)),
                "moving_average_200day": float(data.get("200DayMovingAverage", 0))
            },
            "analyst_metrics": {
                "target_price": float(data.get("AnalystTargetPrice", 0)),
                "ratings": {
                    "strong_buy": int(data.get("AnalystRatingStrongBuy", 0)),
                    "buy": int(data.get("AnalystRatingBuy", 0)),
                    "hold": int(data.get("AnalystRatingHold", 0)),
                    "sell": int(data.get("AnalystRatingSell", 0)),
                    "strong_sell": int(data.get("AnalystRatingStrongSell", 0))
                }
            },
            "price_metrics": {
                "price_to_dividend_ratio": current_price / dividend_per_share if dividend_per_share > 0 else 0,
                "distance_from_50ma": ((current_price / float(data.get("50DayMovingAverage", current_price)) - 1) * 100) if current_price > 0 else 0,
                "distance_from_200ma": ((current_price / float(data.get("200DayMovingAverage", current_price)) - 1) * 100) if current_price > 0 else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
