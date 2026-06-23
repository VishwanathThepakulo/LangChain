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



from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

@tool(description="Answer the queries related to web search")
def websearch(query):
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    print(results)

@tool(description="This tool is used to excute the shell(Terminal)")
def shell(query):
    shell_tool = ShellTool()
    result = shell_tool.invoke(query)
    print(result)

import requests
@tool(description="answer related to weather data(input is only city name)")
def get_weather(city_name):
    api_key = os.getenv('WEATHER_APIKEY')
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {
            'q': city_name,
            'appid': api_key,
            'units': 'metric' 
        }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code==200:
            main = data['main']
            weather = data['weather'][0]
            
            print(f"--- Weather in {data['name']}, {data['sys']['country']} ---")
            print(f"Temperature: {main['temp']}°C")
            print(f"Feels Like : {main['feels_like']}°C")
            print(f"Humidity   : {main['humidity']}%")
            print(f"Condition  : {weather['description'].capitalize()}")
        else:
            print(f"Error: {data.get('message', 'Unable to fetch data.')}")
            
    except Exception as e:
        print(f"An error occurred: {e}")     




def tool_calling():
    model_with_tools = model.bind_tools([python_queries, java_queries, websearch, shell, get_weather])
    ai_message = model_with_tools.invoke("climate in hyderabad")
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
        elif tool_call["name"] == "shell":
            result = shell.invoke(
                tool_call["args"]
            )
        elif tool_call["name"] == "get_weather":
            result = get_weather.invoke(
                tool_call["args"]
            )

tool_calling()

