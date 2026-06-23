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




api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("Api key not found")
model = init_chat_model(
    model='llama-3.3-70b-versatile',
    model_provider='groq',
    api_key=api_key
)

@tool(description="answer the queries related to python only.")
def python_queries(query):
    system_msg = SystemMessage(
        content = "You are a helpfull assistant Answers only question related to python proramming language"
    )
    human_msg = HumanMessage(
        content = f"{query}"
    )
    messages = [system_msg, human_msg]
    response = model.invoke(messages)
    print("============================")
    print(response.content)
    print("============================")
    return response.content

@tool(description="answer the queries related to java only.")
def java_queries(query):
    system_msg = SystemMessage(
        content = "You are a helpfull assistant Answers only question related to java proramming language"
    )
    human_msg = HumanMessage(
        content = f"{query}"
    )
    messages = [system_msg, human_msg]
    response = model.invoke(messages)
    print(response.content)
    return response.content



from langchain_community.tools import DuckDuckGoSearchRun

@tool(description="Answer the queries related to web search")
def websearch(query):
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    print(results)




def tool_calling():
    model_with_tools = model.bind_tools([python_queries, java_queries, websearch])
    ai_message = model_with_tools.invoke("who is modi")
    print("Tool Calls Needed:", ai_message.tool_calls)     
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "python_queries":
            result = python_queries.invoke(
                tool_call["args"]
            )
        elif tool_call["name"] == "java_queries":
            result = java_queries.invoke(
                tool_call["args"]
            )
        elif tool_call["name"] == "websearch":
            result = websearch.invoke(
                tool_call["args"]
            )

tool_calling()

# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# results = search_tool.invoke('Time in hyderabad india now')
# print(results)