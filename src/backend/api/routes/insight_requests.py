from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.data.load.dividend_insights_load import DividendInsightsLoader

# Initialize the APIRouter
router = APIRouter(
    prefix="/insights",
    tags=["insights"]
)

# Pydantic model for the request payload
class InsightsRequest(BaseModel):
    symbol: str
    uploaded_file: Optional[str] = None

# Initialize the loader
OUTPUT_CSV_PATH = "dividend_insights.csv"
loader = DividendInsightsLoader(OUTPUT_CSV_PATH)

# Route for generating insights
@router.post("/generate_insights")
async def generate_insights(request: InsightsRequest):
    """
    Endpoint to generate and retrieve dividend insights after model training.
    """
    try:
        insights = await loader.generate_insights(request.symbol, request.uploaded_file)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")
