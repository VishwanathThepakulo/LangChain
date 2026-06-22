from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
import os
import logging

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    filemode='a',
    filename='app.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)



class ToolsLearning():
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise Exception("Api key not found")
        self.model = init_chat_model(
            model='llama-3.3-70b-versatile',
            model_provider='groq',
            api_key=self.api_key,
            max_tokens = 200
        )

    @tool(description="answer the queries related to python only.")
    def python_queries(self, query):
        system_msg = SystemMessage(
            content = "You are a helpfull assistant Answers only question related to python proramming language"
        )
        human_msg = HumanMessage(
            content = f"{query}"
        )
        messages = [system_msg, human_msg]
        response = self.model.invoke(messages)
        print("============================")
        print(response.content)
        print("============================")
        return response.content
    
    @tool(description="answer the queries related to java only.")
    def java_queries(self, query):
        system_msg = SystemMessage(
            content = "You are a helpfull assistant Answers only question related to java proramming language"
        )
        human_msg = HumanMessage(
            content = f"{query}"
        )
        messages = [system_msg, human_msg]
        response = self.model.invoke(messages)
        print(response.content)
        return response.content
    
    def tool_calling(self):
        model_with_tools = self.model.bind_tools([self.python_queries, self.java_queries])
        ai_message = model_with_tools.invoke("what is python")
        print("Text Content:", ai_message)
        print("Tool Calls Needed:", ai_message.tool_calls)

    
    
        
if __name__ == "__main__":
    tools = ToolsLearning()
    tools.tool_calling()


