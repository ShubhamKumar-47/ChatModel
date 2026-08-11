from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# MODEL
# ============================================================

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# PAGE
# ============================================================

st.title("🤖 Mood Based AI Chatbot")
st.caption("Choose AI personality and start chatting | Type 0 to stop")


# ============================================================
# MODE SELECTION
# ============================================================

mode_choice = st.radio(
    "Choose your AI Mode:",
    ["😡 Angry", "😂 Funny", "😢 Sad"],
    horizontal=True
)


# ============================================================
# MAP MODE TO SYSTEM MESSAGE
# ============================================================

if mode_choice == "😡 Angry":

    mode = (
        "You are an angry AI agent. "
        "You respond aggressively and impatiently. "
        "However, you must still answer the user's question helpfully."
    )

elif mode_choice == "😂 Funny":

    mode = (
        "You are a very funny AI agent. "
        "You respond with humor and jokes while still answering "
        "the user's question correctly."
    )

else:

    mode = (
        "You are a very sad AI agent. "
        "You respond in a depressed and emotional tone, "
        "but you must still answer the user's question helpfully."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(content=mode)
    ]


# ============================================================
# UPDATE SYSTEM MESSAGE WHEN MODE CHANGES
# ============================================================

else:

    st.session_state.messages[0] = SystemMessage(
        content=mode
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# ============================================================
# USER INPUT
# ============================================================

prompt = st.chat_input("Ask anything")


# ============================================================
# CHATBOT
# ============================================================

if prompt:

    # Exit
    if prompt == "0":

        st.write("Exiting chatbot. Goodbye!")

    else:

        # Add user's message
        st.session_state.messages.append(
            HumanMessage(content=prompt)
        )

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from Mistral
        response = model.invoke(
            st.session_state.messages
        )

        # Add AI response
        st.session_state.messages.append(
            AIMessage(content=response.content)
        )

        # Display AI response
        with st.chat_message("assistant"):
            st.write(response.content)