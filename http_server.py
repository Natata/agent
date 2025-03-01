# Flask imports for the HTTP server
from flask import Flask, request, session
from agent import Agent, MODEL_LLAMA31
import uuid

# Flask application setup
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a secure secret key in production
agent = Agent(model=MODEL_LLAMA31)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Handle POST requests to the /chatbot endpoint."""
    data = request.get_json()
    message = data.get("message")
    if not message:
        return {"error": "No message provided"}, 400
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session["thread_id"]}}
    response = agent.invoke(message, config)
    return {"response": response}

# Run the server
if __name__ == "__main__":
    app.run(debug=True, port=5566)