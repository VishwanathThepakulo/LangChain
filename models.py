from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider='groq',
    api_key = api_key,
    temperature=0.7,
    timeout=30,
    max_tokens=100,
    max_retries=6,
)

response = model.invoke("why do parrot talk?")
print(response.content)


