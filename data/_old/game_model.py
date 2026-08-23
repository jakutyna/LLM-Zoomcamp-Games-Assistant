from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class GameGenre(str, Enum):
    ACTION = "Action"
    ADVENTURE = "Adventure"
    RPG = "RPG"
    RTS = "RTS"
    SHOOTER = "Shooter"
    SPORTS = "Sports"
    SURVIVAL = "Survival"

class Game(BaseModel):
    rank: int = Field(description="Rank of the game in the all-time best ranking; also used as game id")
    title: str = Field(description="Game title")
    genre: GameGenre = Field(description="Genre of the game")
    platforms: List[str] = Field(description="Platform(s) the game is available on, e.g. 'PC', 'Console', 'Mobile', etc.")
    release_date: str = Field(description="Release date of the game in YYYY-MM-DD format")
    developer: str = Field(description="Developer of the game")
    publisher: str = Field(description="Publisher of the game")
    description: str = Field(description="Brief description of the game")
