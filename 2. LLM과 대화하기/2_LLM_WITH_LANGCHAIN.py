from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B-MLX-8bit",
    base_url="http://172.30.1.42:8000/v1",
    stream_usage=False,
    api_key="..."
)

res = llm.invoke([
    HumanMessage("한국어로 반갑게 인사해줘? /no_think")
])


print(res.content)