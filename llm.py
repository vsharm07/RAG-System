import ollama

def call_llm(prompt):
    response = ollama.generate(model="llama3", prompt=prompt)
    return response["response"] 