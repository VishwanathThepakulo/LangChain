from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv('GROQ_API_KEY')


model = init_chat_model(
    model='llama-3.3-70b-versatile',
    model_provider='groq',
    api_key=api_key,
    max_tokens = 200
)

messages = [
    SystemMessage(
        content="You are a helpful assistant."
    )
]

while True:
    user_input = input("Enter your query : ")
    if user_input.lower() == 'exit':
        break
    messages.append(
        HumanMessage(content=user_input)
    )
    response = model.invoke(messages) 
    messages.append(response)
    print(response.content)
    print(f"=========================\n{messages}")