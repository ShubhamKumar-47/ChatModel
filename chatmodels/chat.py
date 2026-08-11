from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gemini-3-flash-preview",
    model_provider="google_genai"
)

response = model.invoke("Hello, how are you?")
print(response.content)