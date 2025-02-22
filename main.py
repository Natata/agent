from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(
    model="llama3",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to German."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chains = prompt | llm
resp = chains.invoke(
    {
        "messages": [
            HumanMessage(content="I love programming."),
        ],
    }
)

print(resp.content)
