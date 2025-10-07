from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


llm = ChatOllama(model="llama3.2", temperature=0)
agent = create_react_agent(
    model=llm,
    prompt=(
        """You are a helpful assistant that does your best to help the user as much as you can. Answer every question with utmost honesty and complete knowledge. If you don't know the answer, just say you don't know. Never make up an answer."""
    ),
    tools=[],
)
messages = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append({"role": "user", "content": user_input})

    results = agent.invoke({"messages": messages})
    ai_messages = [
        m
        for m in results["messages"]
        if getattr(m, "role", None) == "assistant"
        or m.__class__.__name__ == "AIMessage"
    ]
    if ai_messages:
        assistant_reply = ai_messages[-1].content
    else:
        assistant_reply = "No assistant reply found."
    messages.append({"role": "assistant", "content": assistant_reply})
    print("Assistant:", assistant_reply)
