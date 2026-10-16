import os
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Hermes Agent - Gemma 3 1B")

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=os.getenv("OLLAMA_API_KEY")
)

class ChatRequest(BaseModel):
    prompt: str
    system: str = "Tu es Hermes Agent, un assistant expert en automatisation. Réponds en français de façon concise et utile."

@app.get("/")
def home():
    return {"status": "live", "model": "gemma3:1b"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not os.getenv("OLLAMA_API_KEY"):
        raise HTTPException(status_code=500, detail="OLLAMA_API_KEY manquante")
    try:
        completion = client.chat.completions.create(
            model="gemma3:1b",
            messages=[{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
            temperature=0.7
        )
        return {"model": "gemma3:1b", "response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
