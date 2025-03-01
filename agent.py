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

# Import the CustomerServiceClient and related classes
from fake_client import CustomerServiceClient, Customer, UserNotFoundError
from uuid import UUID

MODEL_LLAMA31 = "llama3.1"
NODE_TOOL_CULCULATOR = "node_tool_calculator"
NODE_TOOL_RANDOM_NUMBER = "node_tool_random_number"
NODE_CUSTOMER_SERVICE = "node_customer_service"

# -- tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def random_number() -> int:
    """Return a random number."""
    return 123123

# Instantiate the CustomerServiceClient globally so it persists across calls
customer_client = CustomerServiceClient()

@tool
def create_customer(name: str) -> dict:
    """Create a new customer with the given name."""
    customer = customer_client.create_customer(name)
    return {"id": str(customer.id), "name": customer.name}

@tool
def get_customer(customer_id: str) -> dict:
    """Retrieve a customer by ID."""
    try:
        customer = customer_client.get_customer(UUID(customer_id))
        return {"id": str(customer.id), "name": customer.name}
    except UserNotFoundError as e:
        return {"error": str(e)}

@tool
def list_customers(offset: int, limit: int) -> dict:
    """List customers with pagination. Limit must be 1, 2, or 3."""
    try:
        result = customer_client.list_customers(offset, limit)
        return {
            "customers": [{"id": str(c.id), "name": c.name} for c in result["customers"]],
            "offset": result["offset"],
            "limit": result["limit"]
        }
    except ValueError as e:
        return {"error": str(e)}

@tool
def update_customer(customer_id: str, name: str) -> dict:
    """Update a customer's name by ID."""
    try:
        customer = Customer(id=UUID(customer_id), name=name)
        updated_customer = customer_client.update_customer(customer)
        return {"id": str(updated_customer.id), "name": updated_customer.name}
    except UserNotFoundError as e:
        return {"error": str(e)}

@tool
def delete_customer(customer_id: str) -> dict:
    """Delete a customer by ID."""
    try:
        customer_client.delete_customer(UUID(customer_id))
        return {"success": f"Customer with id {customer_id} deleted."}
    except UserNotFoundError as e:
        return {"error": str(e)}


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# -- condition edges

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

def customer_service_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["node_customer_service", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        tool_name = ai_message.tool_calls[-1]['name']
        if tool_name in ["create_customer", "get_customer", "list_customers", "update_customer", "delete_customer"]:
            return NODE_CUSTOMER_SERVICE
    return "__end__"

class Agent:
    def __init__(self, model=MODEL_LLAMA31):
        tools = [
            multiply,
            random_number,
            create_customer,
            get_customer,
            list_customers,
            update_customer,
            delete_customer
        ]

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
        tool_node_customer_service = ToolNode(
            tools=[create_customer, get_customer, list_customers, update_customer, delete_customer],
            name=NODE_CUSTOMER_SERVICE
        )
        graph_builder.add_node(NODE_CUSTOMER_SERVICE, tool_node_customer_service)
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
        graph_builder.add_conditional_edges(
            "assistant", 
            customer_service_tools_condition,
        )
        graph_builder.add_edge(NODE_CUSTOMER_SERVICE, "assistant")

        graph_builder.set_entry_point("assistant")
        graph_builder.set_finish_point("assistant")

        # memory
        memory = MemorySaver()        

        self.graph = graph_builder.compile(checkpointer=memory)

        ## NOTE: comment out to show the graph
        # try:
        #     display(Image(self.graph.get_graph().draw_mermaid_png()))
        # except Exception:
        #     # This requires some extra dependencies and is optional
        #     pass

    def assistant(self, state: State):
        system_message = SystemMessage(
            content="""
                You are a helpful assistant. 
                Use the 'multiply' tool only when the user explicitly asks for multiplication (e.g., 'multiply 3 and 4' or 'what is 3 times 4'). 
                Use the 'random_number' tool only when the user explicitly asks for a random number (e.g., 'give me a random number').
                Use the customer service tools ('create_customer', 'get_customer', 'list_customers', 'update_customer', 'delete_customer') when the user requests customer-related actions (e.g., 'create a customer named John', 'list customers', 'update customer id xyz to name Alice', etc.).
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