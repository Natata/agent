from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    print(f"Multiplying {a} and {b}")
    return a * b

# constants llama3.1
MODEL_LLAMA31 = "llama3.1"

class AgentApp:
    def __init__(self, model=MODEL_LLAMA31):
        self.llm = ChatOllama(
            model=model,
            temperature=0,
        )

        tool = ChatOllama(
            model=model,
            temperature=0,
        ).bind_tools([multiply])

        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_node("multiply", tool)

        self.workflow.add_edge(START, "model")
        self.workflow.add_conditional_edges
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    # Define the function that calls the model
    def call_model(self, state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = self.llm.invoke(messages)

        # debug
        # print("#####", response.content)

        return {"messages": response}

    def invoke(self, thread_id, message) -> str:
        output = self.app.invoke(
            {
                "messages": [
                    HumanMessage(content=message)
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        print("#####", output)
        print(output["messages"][-1].tool_calls)
        return output["messages"][-1].content
