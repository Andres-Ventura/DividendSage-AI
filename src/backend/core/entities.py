from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

class BaseEntity:
    """Base class for all entities with common attributes"""
    def __init__(self):
        self.id: UUID = uuid4()
        self.created_at: datetime = datetime.now(datetime.UTC)
        self.updated_at: datetime = datetime.now(datetime.UTC)

class User(BaseEntity):
    """User entity representing application users"""
    def __init__(
        self,
        email: str,
        username: str,
        password_hash: str,
        full_name: Optional[str] = None,
        is_active: bool = True
    ):
        super().__init__()
        self.email = email
        self.username = username
        self.password_hash = password_hash
        self.full_name = full_name
        self.is_active = is_active

class Post(BaseEntity):
    """Post entity representing user-created content"""
    def __init__(
        self,
        title: str,
        content: str,
        author_id: UUID,
        tags: List[str] = None
    ):
        super().__init__()
        self.title = title
        self.content = content
        self.author_id = author_id
        self.tags = tags or []