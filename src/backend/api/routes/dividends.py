from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import APIRouter, Depends

from backend.core.services import UserService


load_dotenv()

router = APIRouter(
    prefix="/dividends",
    tags=["dividends"],
    # dependencies=[Depends()]
)
