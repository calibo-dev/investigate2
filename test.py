# # pip install -qU langchain "langchain[openai]"
# from langgraph.prebuilt import create_agent

# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

# agent = create_agent(
#     model="openai:gpt-5.5",
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# )

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
# )
# print(result["messages"][-1].content_blocks)

import langchain
import langgraph.prebuilt
print(dir(langgraph.prebuilt))
print(langchain.__version__)

# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent

print(dir(langchain.agents))

