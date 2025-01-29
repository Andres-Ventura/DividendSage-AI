from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class FinancialDataProvider(ABC):
    """Abstract base class for financial data providers"""
    
    @abstractmethod
    async def get_dividend_history(self, symbol: str) -> Dict[str, Any]:
        """Get historical dividend data for a symbol"""
        pass
    
    @abstractmethod
    async def get_daily_adjusted(self, symbol: str, days: int) -> Dict[str, Any]:
        """Get daily adjusted price data"""
        pass
    
    @abstractmethod
    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Get cash flow statements"""
        pass

    @abstractmethod
    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """Get earnings data"""
        pass

    @abstractmethod
    async def get_company_stats(self, symbol: str) -> Dict[str, Any]:
        """Get company overview and statistics"""
        pass