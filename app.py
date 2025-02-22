from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

class AgentApp:
    def __init__(self, model="llama3"):
        self.llm = ChatOllama(
            model=model,
            temperature=0,
        )
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
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
        # print(response.content)

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

        return output["messages"][-1].content
    
if __name__ == "__main__":
    app = AgentApp(model="llama3")

    # loop to get user input and call the model
    while True:
        message = input(">>>: ")
        if message.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = app.invoke(thread_id="1", message=message)
        print(f"Chatbot: {response}")