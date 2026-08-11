from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

message = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    print("------------------- Welcome! Type 0 to exit the application -------------------")
    prompt = input("You: ")
    message.append(HumanMessage(content=prompt))
    if prompt == "0":
        print("Exiting chatbot. Goodbye!")
        break

    response = model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("Bot:", response.content)