from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()


# ---------------- MODEL ----------------

model = ChatMistralAI(
    model="mistral-small-2506"
)


# ---------------- PYDANTIC MODEL ----------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# ---------------- OUTPUT PARSER ----------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a movie information extraction assistant.

Extract movie information from the given paragraph.

{format_instructions}

If a field is not mentioned in the paragraph:
- Use null for optional fields.
- Use an empty list [] for list fields.

Do not add any information that is not present in the paragraph.
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])


# ---------------- USER INPUT ----------------

para = input("Give your paragraph: ")


# ---------------- CREATE FINAL PROMPT ----------------

final_prompt = prompt.invoke({
    "paragraph": para,
    "format_instructions": parser.get_format_instructions()
})


# ---------------- MODEL RESPONSE ----------------

response = model.invoke(final_prompt)


# ---------------- PARSE RESPONSE ----------------

movie_data = parser.parse(response.content)


# ---------------- OUTPUT ----------------

print("\n------------- MOVIE DATA -------------")

print("Title:", movie_data.title)
print("Release Year:", movie_data.release_year)
print("Genre:", movie_data.genre)
print("Director:", movie_data.director)
print("Cast:", movie_data.cast)
print("Rating:", movie_data.rating)
print("Summary:", movie_data.summary)