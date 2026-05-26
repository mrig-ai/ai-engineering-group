from pydantic import BaseModel, Field


class QueryExpansion(BaseModel):
    questions: list[str] = Field(
        min_items=3,
        max_items=3,
        description=(
            "A list of exactly 3 **DISTINCT** search questions: "
            "1. Literal: The user question translated to English (or spelling-fixed). "
            "2. SEO: The core keywords only, removing filler words. "
            "3. Variation: A version using synonyms, converting symbols like '+' or '&' into words (e.g., 'Plus', 'and')."
        ),
    )
    detected_language: str = Field(
        description="The language name the user is speaking (e.g., German, English, French, Spanish, Portuguese)."  # noqa: E501
    )
