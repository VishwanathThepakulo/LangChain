from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()



class Joke(BaseModel):
    setup : str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")


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
    configurable_fields=("model", "model_provider", "temperature", 'max_tokens')
)
print("=================Response 1====================")
response1 = configurable_model.invoke("why do parrot talk?")
print(response1.content)

print("=================Response 2====================")

model2 = configurable_model.with_config(
    configurable={
        'model':"llama-3.3-70b-versatile", 
        'model_provider':'groq',
        'temperature':0.5,                  
        'max_tokens':200
    }
)

structured_output = model2.with_structured_output(Joke, method="json_mode")

prompt = [
    ("system", (
        "You are a funny assistant. You must respond strictly in JSON format.\n"
        "Your JSON object MUST contain exactly these two keys and nothing else:\n"
        "- 'setup': The setup of the joke\n"
        "- 'punchline': The punchline"
    )),
    ("human", "tell me a joke about china")
]

response2 = structured_output.invoke(prompt)
print(response2)


