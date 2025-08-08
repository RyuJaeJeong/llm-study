from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from repository import reset_store
from dotenv import load_dotenv
from icecream import ic
import os
import sqlite3

load_dotenv()
langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"]
)

langfuse_handler = CallbackHandler()
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
vector_store = reset_store()
graph_builder = StateGraph(MessagesState)

@tool
def calculate(expression: str):
    """Calculate mathematical expressions."""
    result = eval(expression)
    return f"""Result: {result}"""

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def get_now(format: str):
    """현재 시간을 조회 합니다."""

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve, calculate])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve, calculate])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
           or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=checkpointer)

input_message = "7곱하기 3 더하기 2 빼기 100은"
for step in graph.stream(
{"messages": [{"role": "user", "content": input_message}]},
    config={"configurable": {"thread_id": "1"}, "callbacks": [langfuse_handler]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# png_data = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_data)
#
# print(f"그래프가 graph.png로 저장되었습니다.")