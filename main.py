from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0,
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
    ("system", "J\'adore le programmation.\n\n(Note: \"programming\" can also be translated as \"informatique\" or \"programmation\", but in this context, I chose to use the more informal and conversational translation \"le programmation\")"),
    ("human", "Have a nice trip")
]
resp = llm.invoke(messages)
print(resp.content)
