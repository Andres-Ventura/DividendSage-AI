from fastapi import APIRouter, HTTPException
import requests
from config import ALPHA_VANTAGE_API_KEY

router = APIRouter(
    prefix="/api/health",
    tags=["health"]
)

@router.get("")
async def check_health():
    """
    Performs health checks on the API and its dependencies
    """
    health_status = {
        "status": "healthy",
        "services": {
            "alpha_vantage_api": check_alpha_vantage_status()
        }
    }

    # If any service is unhealthy, change overall status
    if not all(service["status"] == "healthy" for service in health_status["services"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status

def check_alpha_vantage_status():
    """
    Checks if Alpha Vantange API is accessible using a simple request
    """
    try:
        # Use a simple request to test API availablility
        test_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(test_url, timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "latency_ms": int(response.elapsed.total_seconds() * 1000) 
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"Alpha Vantage API returned status code {response.status_code}"
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }