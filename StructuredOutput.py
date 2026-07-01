# from langchain.chat_models import init_chat_model
# from dotenv import load_dotenv
# import os
# from pydantic import BaseModel

# load_dotenv()

# class Person(BaseModel):
#     name: str
#     age: str
    
    
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise Exception("Api key not found")
# model = init_chat_model(
#     model='meta-llama/llama-4-scout-17b-16e-instruct',
#     model_provider='groq',
#     api_key=api_key
# )
    
# structured_model = model.with_structured_output(Person)

# result = structured_model.invoke("John is 30 years old")
# print(result.name)
# print(type(result))
# results = result.model_dump()
# print(results['age'])
# print(type(results))


from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    raise Exception("Api key not found")


class ContactInfo(BaseModel):
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

model = init_chat_model(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq",
    api_key=api_key
)


agent = create_agent(
    model=model,
    response_format=ContactInfo
)


result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])

