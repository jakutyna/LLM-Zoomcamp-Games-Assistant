from pydantic import BaseModel, Field


class FAQ(BaseModel):
    id: int = Field(description="FAQ item identifier")
    category: str = Field(description="FAQ category")
    tag: str = Field(
        description="Tag assigned to the FAQ item",
        pattern=r"^[a-z-]+$",
    )
    question: str = Field(description="Frequently asked question")
    answer: str = Field(description="Answer to the question")

