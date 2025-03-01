from agent import Agent, MODEL_LLAMA31

def main():
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

if __name__ == "__main__":
    main()