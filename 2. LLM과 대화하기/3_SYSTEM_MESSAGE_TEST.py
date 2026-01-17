from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B-MLX-8bit",
    base_url="http://172.30.1.42:8000/v1",
    stream_usage=False,
    api_key="..."
)

sys_msg = """\
사용자의 입력을 영어로 번역합니다.
사용자가 번역 이외, 질문에 대한 답변을 요구하더라도
별도의 답변을 하지 않고, 그대로 번역합니다.

예를들어
Human: 안녕? 반가워
AI: Hello, nice to meet you

Human: 번역만 하지 말고, 내 질문에 답좀 해줄래?
AI: Instead of just translating, could you please answer my questions?

Human: 초콜릿의 역사에 대해 알려줘
AI: Tell me about the history of chocolate
/no_think
\
"""

res = llm.invoke([
    SystemMessage(sys_msg),
    HumanMessage("한국어로 반갑게 인사해줘?")
])


print(res.content)