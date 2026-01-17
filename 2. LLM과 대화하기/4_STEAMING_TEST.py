from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B-MLX-8bit",
    base_url="http://172.30.1.42:8000/v1",
    stream_usage=True,
    api_key="..."
)

def chat_stream(query:str):
    for chunk in llm.stream(query):
        print
        yield chunk.content

for chunk in chat_stream("간단한 자기소개 부탁해 /no_think"):
    print(chunk, end="", flush=True)