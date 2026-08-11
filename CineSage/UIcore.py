import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI


# -------------------- Setup --------------------

load_dotenv()


@st.cache_resource
def get_model():
    return ChatMistralAI(
        model="mistral-small-2506"
    )


model = get_model()


# -------------------- Schema --------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# -------------------- Parser --------------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# -------------------- Prompt --------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a movie information extraction assistant.

Extract movie information from the paragraph.

{format_instructions}

If information is not available:
- Use null for optional fields.
- Use [] for list fields.

Do not invent information.
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])


# -------------------- Page Configuration --------------------

st.set_page_config(
    page_title="🎬 Movie Info Extractor",
    page_icon="🎬",
    layout="centered"
)


# -------------------- UI --------------------

st.title("🎬 Movie Information Extractor")

st.write(
    "Paste any movie description and AI will convert it "
    "into structured data."
)


paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=200,
    placeholder="Example: Interstellar is a science fiction film..."
)


# -------------------- Extract Button --------------------

if st.button("Extract Data"):

    if not paragraph.strip():

        st.warning("Please enter a paragraph first.")

    else:

        with st.spinner("Analyzing movie..."):

            try:

                # Create final prompt
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })


                # Get response from Mistral
                response = model.invoke(final_prompt)


                # ---------------- RAW OUTPUT ----------------

                st.subheader("Raw Model Output")

                st.code(
                    response.content,
                    language="json"
                )


                # ---------------- PARSE OUTPUT ----------------

                movie_data = parser.parse(
                    response.content
                )


                # ---------------- STRUCTURED OUTPUT ----------------

                st.subheader("Structured Output")

                st.json(
                    movie_data.model_dump()
                )


                st.success(
                    "Extraction Completed Successfully!"
                )


            except Exception as e:

                st.error(
                    "Failed to parse response. "
                    "Model did not follow the schema."
                )

                st.exception(e)