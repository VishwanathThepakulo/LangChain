
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
import os
import requests
from dotenv import load_dotenv
load_dotenv()



api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise Exception("LLM API key not found")

model = init_chat_model(
    model='meta-llama/llama-4-scout-17b-16e-instruct',
    model_provider='groq',
    api_key=api_key
)




@tool
def get_currency_tool(base_currency: str, target_currency:str)->float:
    """
    This function fetch currency factor between a given currency and a target currency
    """
    rate_api = os.getenv('Exchange_Rate_API')
    if not rate_api:
        raise Exception("Exchange_Rate_API not found")
    url = f"https://v6.exchangerate-api.com/v6/{rate_api}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    # print(response.json())
    return response.json()

# result =  get_currency('USD', 'INR')
# print(result['conversion_rate'])



@tool
def converter_tool(base_currency_value: float, conversion_rate: float)-> float:
    """
    given a currency conversation rate this function calculates the target currency value from a given base currency value
    """
    return  base_currency_value * conversion_rate
# results = converter(10, 94.4628)
# print(results)


tools_map = {
    'get_currency_tool':get_currency_tool,
    'converter_tool':converter_tool
}

messages = [SystemMessage(content="You are a helpfull assistant")]

model_with_tool = model.bind_tools(list(tools_map.values()))
# print("===============================================================")
# # print(f'\n\n{model_with_tool}\n\n')
# print("===============================================================")



print("Assistant is ready! Type 'exit' to quit.")



# print(f"\n\n\n===> {ai_response['name']}")
# print(f"\n\n\n===> {ai_response['args']}")
# print(f"\n\n\n===> {ai_response['id']}")
# print(f"\n\n\n===> {ai_response['type']}")





# while True:
#     user_query = input('\nQuery : ')
#     if user_query.lower() in ["exit", "quit"]:
#         break
#     messages.append(HumanMessage(content=user_query))

#     ai_response = model_with_tool.invoke(messages)
#     messages.append(ai_response)
#     # ai_response = ai_tool_message.tool_calls[0]

#     if ai_response.tool_calls:
#         for tool_call in ai_response.tool_calls:
#             tool_name = tool_call["name"]
#             tool_args = tool_call["args"]
#             tool_id = tool_call["id"]
#             # print(f'tool call name ====> {tool_call["name"]}')
#             # print(f'\n\ntool keys ===> {tools_map.keys()}')
#             print(f'Executing tool: {tool_name} with args: {tool_args}')
#             tool_func = tools_map[tool_name]
#             tool_result = tool_func.invoke(tool_args)
#             print("\n\n")
#             # print(tool_result.get('conversion_rate'))
#             messages.append(
#                             ToolMessage(
#                                 content=str(tool_result),
#                                 tool_call_id=tool_id
#                             )
#                         )
#         final_response = model_with_tool.invoke(messages)
#         messages.append(final_response)
#         print(final_response.content)
#     else:
#         print(f"\nAssistant: {ai_response.content}")

while True:
    user_query = input('\nQuery : ')
    if user_query.lower() in ["exit", "quit"]:
        break
        
    messages.append(HumanMessage(content=user_query))

    # Keep looping until the model gives a final text response without calling tools
    while True:
        ai_response = model_with_tool.invoke(messages)
        messages.append(ai_response)

        # If the model wants to call tools, execute them and stay in the loop
        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f'Executing tool: {tool_name} with args: {tool_args}')

                tool_func = tools_map[tool_name]
                tool_result = tool_func.invoke(tool_args)
                
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    )
                )
            # Continue the inner loop so the model can evaluate the tool output
            continue 
        
        # If no tools were called, it's a final response. Print it and break the inner loop.
        else:
            print(f"\nAssistant: {ai_response.content}")
            break