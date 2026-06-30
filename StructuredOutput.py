from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

class Person(BaseModel):
    name: str
    age: str
    
    
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("Api key not found")
model = init_chat_model(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    model_provider='groq',
    api_key=api_key
)
    
structured_model = model.with_structured_output(Person)

result = structured_model.invoke("John is 30 years old")
print(result)
