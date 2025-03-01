from typing import (
    Any,
    Literal, 
    Annotated,
    Union,
)

from typing_extensions import TypedDict

from pydantic import BaseModel

from IPython.display import Image, display

from langchain_ollama import ChatOllama

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage,
    AnyMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


MODEL_LLAMA31 = "llama3.1"
NODE_TOOL_CULCULATOR = "node_tool_calculator"
NODE_TOOL_RANDOM_NUMBER = "node_tool_random_number"

# -- tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def random_number() -> int:
    """Return a random number."""
    return 123123

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def multiple_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["node_tool_calculator", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("**** ai_message.tool_calls:", ai_message.tool_calls)
        if ai_message.tool_calls[-1]['name'] == "multiply":
            return NODE_TOOL_CULCULATOR
    return "__end__"

def random_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["node_tool_random_number", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("**** ai_message.tool_calls:", ai_message.tool_calls)
        if ai_message.tool_calls[-1]['name'] == "random_number":
            return NODE_TOOL_RANDOM_NUMBER
    return "__end__"

class Agent:
    def __init__(self, model=MODEL_LLAMA31):
        tools = [multiply, random_number]

        self.llm = ChatOllama(
            model=model,
            temperature=0,
        ).bind_tools(tools)

        graph_builder = StateGraph(State)
        # nodes
        graph_builder.add_node("assistant", self.assistant)
        tool_node_multiple = ToolNode(tools=[multiply], name="node_tool_calculator")
        graph_builder.add_node(NODE_TOOL_CULCULATOR, tool_node_multiple)
        tool_node_random_number = ToolNode(tools=[random_number], name="node_tool_random_number")
        graph_builder.add_node(NODE_TOOL_RANDOM_NUMBER, tool_node_random_number)
        # edges
        graph_builder.add_conditional_edges(
            "assistant",
            multiple_tools_condition,
        )
        graph_builder.add_edge(NODE_TOOL_CULCULATOR, "assistant")
        graph_builder.add_conditional_edges(
            "assistant",
            random_tools_condition,
        )
        graph_builder.add_edge(NODE_TOOL_RANDOM_NUMBER, "assistant")

        graph_builder.set_entry_point("assistant")
        graph_builder.set_finish_point("assistant")

        # memory
        memory = MemorySaver()        

        self.graph = graph_builder.compile(checkpointer=memory)

        ## NOTE: comment out to show the graph
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass

    def assistant(self, state: State):
        system_message = SystemMessage(
            content="""
                You are a helpful assistant. 
                Use the 'multiply' tool only when the user explicitly asks for multiplication (e.g., 'multiply 3 and 4' or 'what is 3 times 4'). 
                Use the 'random_number' tool only when the user explicitly asks for a random number (e.g., 'give me a random number').
                For all other questions, answer directly using your knowledge.
                """,
        )
        messages = state["messages"]
        response = self.llm.invoke([system_message] + messages)
        return {"messages": response}

    def invoke(self, user_input: str, config: dict = {}) -> str:
        response = ""
        for event in self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        ):
            print("### debug event:", event)
            for value in event.values():
                # value[-1].pretty_print()
                response = value[-1].content
        return response

    def state(self, config: dict = {}):
        return self.graph.get_state(config)

if __name__ == "__main__":
    agent = Agent(model=MODEL_LLAMA31)
    
    # loop to get user input and call the model
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input(">>>: ")
        match user_input.lower():
            case "exit":
                print("Chatbot: Goodbye!")
                break
            case "state":
                print(agent.state(config))
            case "":
                print("Chatbot: Please type something.")
            case _:
                response = agent.invoke(user_input=user_input, config=config)
                print(f"Chatbot: {response}")
                print('-----------------')