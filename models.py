from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

# model = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider='groq',
#     api_key = api_key,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=100,
#     max_retries=6,
# )


configurable_model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider="groq",
    api_key = api_key,
    max_tokens = 200,
    configurable_fields=("model", "model_provider", "temperature")
)
print("=================Response 1====================")
response1 = configurable_model.invoke("why do parrot talk?")
print(response1.content)

print("=================Response 2====================")

model2 = configurable_model.with_config(
    configurable={
        'model':"openai/gpt-oss-120b", 
        'model_provider':'groq',
        'api_key':api_key,
        'temperature':0.5,                  
        'max_tokens':200,
        'max_retries':4,
    }
)


response2 = model2.invoke("what is the capital of sun")
print(response2.content)


