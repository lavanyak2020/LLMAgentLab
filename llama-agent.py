import requests

def query_ollama(prompt):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama3.2',
            'prompt': prompt,
            'stream': False
        }
    )
    data = response.json()
    return data['response']

# Example
reply = query_ollama("What is the capital of France?")
print(reply)
