from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
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
    model='meta-llama/llama-4-scout-17b-16e-instruct',
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
    return results

@tool(description="This tool is used to excute the shell(Terminal)")
def shell(query):
    shell_tool = ShellTool()
    result = shell_tool.invoke(query)
    print(result)
    return result

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
            return data
        else:
            print(f"Error: {data.get('message', 'Unable to fetch data.')}")
            
    except Exception as e:
        print(f"An error occurred: {e}")     



messages = [
    SystemMessage(
        content="You are a helpful assistant."
    )
]

# def tool_calling():
# while True:
#     model_with_tools = model.bind_tools([python_queries, java_queries, websearch, shell, get_weather])
#     user_query = input('Query : ')
#     ai_message = model_with_tools.invoke(user_query)
#     print("Tool Calls Needed:", ai_message.tool_calls)     
#     for tool_call in ai_message.tool_calls:
#         if tool_call["name"] == "python_queries":
#             result = python_queries.invoke(
#                 tool_call["args"]
#             )
#         elif tool_call["name"] == "java_queries":
#             result = java_queries.invoke(
#                 tool_call["args"]
#             )
#         elif tool_call["name"] == "websearch":
#             result = websearch.invoke(
#                 tool_call["args"]
#             )
#         elif tool_call["name"] == "shell":
#             result = shell.invoke(
#                 tool_call["args"]
#             )
#         elif tool_call["name"] == "get_weather":
#             result = get_weather.invoke(
#                 tool_call["args"]
#             )
#     messages.append(
#         HumanMessage(content=user_query)
#     )
#     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#     response = model.invoke(messages) 
#     messages.append(response)
#     print(response.content)
#     print("--------------------------------------------------------------------------------------------------")
    # print(f"=========================\n{messages}")

# tool_calling()

# Create a dictionary to map tool names directly to their functions
tools_map = {
    "python_queries": python_queries,
    "java_queries": java_queries,
    "websearch": websearch,
    "shell": shell,
    "get_weather": get_weather
}

# Bind the tools outside the loop so it only runs once
model_with_tools = model.bind_tools(list(tools_map.values()))

print("Assistant is ready! Type 'exit' to quit.")

while True:
    user_query = input('\nQuery: ')
    if user_query.lower() in ['exit', 'quit']:
        break
        
    # 1. Append the user message to history
    messages.append(HumanMessage(content=user_query))
    
    # 2. Ask the model what to do (it checks history + binds tools)
    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)  # Add the model's intent to history
    
    print("Tool Calls Needed:", ai_message.tool_calls)    
    
    # 3. Handle tool execution if the model requests it
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            
            if tool_name in tools_map:
                # Dynamically invoke the correct tool using its args
                tool_result = tools_map[tool_name].invoke(tool_call["args"])
                
                # Crucial step: Append the ToolMessage so the LLM can read the result
                messages.append(ToolMessage(
                    content=str(tool_result), 
                    tool_call_id=tool_call["id"]
                ))
            else:
                print(f"Error: Model tried to call an unknown tool: {tool_name}")
        
        # 4. Generate the final response based on the newly added tool results
        print("================================== Final Response ==================================")
        final_response = model_with_tools.invoke(messages)
        messages.append(final_response)
        print(final_response.content)
        print("------------------------------------------------------------------------------------")
        print(messages)
    else:
        # If no tools were needed, the AI message already contains the answer
        print("================================== Response ==================================")
        print(ai_message.content)
        print("------------------------------------------------------------------------------")