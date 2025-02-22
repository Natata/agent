from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

llm = ChatOllama(
    model="llama3",
    temperature=0,
)

# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    # debug
    print(response.content)

    return {"messages": response}

# -- Define the workflow

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

resp = app.invoke(
    {
        "messages": [
            HumanMessage(content="What is the capital of Germany?")
        ]
    },
    config={"configurable": {"thread_id": "1"}},
)

app.invoke(
    {
        "messages": [
            HumanMessage(content="What did I just ask you?")
        ]
    },
    config={"configurable": {"thread_id": "1"}},
)
