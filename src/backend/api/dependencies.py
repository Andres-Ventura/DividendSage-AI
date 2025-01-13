from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..services import auth_service

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def get_db():
    """
    Dependency for getting database session.
    Ensures proper handling of database connections with context management.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)]
):
    """
    Dependency for getting the current authenticated
    user. Validates JWT token and returns the user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = await auth_service.get_current_user(token, db)
    if not user:
        raise credentials_exception
    return user

DB = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[dict, Depends(get_current_user)]