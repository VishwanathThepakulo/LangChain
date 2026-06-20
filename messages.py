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

system_msg = SystemMessage("You are a helpful assistan in maths and dont reply to any message apart from math question. ")
human_msg = HumanMessage("what is the capital of india")

messages = [system_msg, human_msg]
response = model.invoke(messages) 
print(response)