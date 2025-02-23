from typing import Annotated

from typing_extensions import TypedDict

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode, tools_condition

from IPython.display import Image, display

MODEL_LLAMA31 = "llama3.1"

# -- tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    # print(f"Multiplying {a} and {b}")
    return a * b

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
class Agent:
    def __init__(self, model=MODEL_LLAMA31):
        self.llm = ChatOllama(
            model=model,
            temperature=0,
        ).bind_tools([multiply])

        graph_builder = StateGraph(State)
        # nodes
        graph_builder.add_node("assistant", self.assistant)
        tool_node = ToolNode(tools=[multiply])
        graph_builder.add_node("tools", tool_node)
        # edges
        #graph_builder.add_edge(START, "assistant")
        #graph_builder.add_edge("assistant", END)
        
        graph_builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        graph_builder.add_edge("tools", "assistant")
        graph_builder.set_entry_point("assistant")
        graph_builder.set_finish_point("assistant")
        self.graph = graph_builder.compile()

        # comment out to show the graph
        # try:
        #     display(Image(self.graph.get_graph().draw_mermaid_png()))
        # except Exception:
        #     # This requires some extra dependencies and is optional
        #     pass

    def assistant(self, state: State):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": response}

    def invoke(self, user_input: str):
        response = ""
        for event in self.graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                # print("### debug:", value)
                if isinstance(value["messages"], list):
                    response = value["messages"][-1].content
                else:
                    response = value["messages"].content
                # if "messages" in value:
                #     response += value["messages"].content
                #     #print("# debug:", value["messages"].content)
        return response

if __name__ == "__main__":
    agent = Agent(model=MODEL_LLAMA31)
    
    # loop to get user input and call the model
    while True:
        user_input = input(">>>: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = agent.invoke(user_input=user_input)
        print(f"Chatbot: {response}")
        print('-----------------')