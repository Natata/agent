from app import AgentApp, MODEL_LLAMA31

def main():
    app = AgentApp(model=MODEL_LLAMA31)

    # loop to get user input and call the model
    while True:
        message = input(">>>: ")
        if message.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = app.invoke(thread_id="1", message=message)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()