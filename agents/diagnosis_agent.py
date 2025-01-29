from typing import Dict, Any
from .base_agent import BaseAgent

class DiagnosisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Diagnosis",
            instructions="""Diagnose an illness based on the given list of symptoms.
            Use the provided context to help formulate an answer.
            Provide output in a clear, structured format."""
        )
    
    async def run(self, ollama_model: str, context: str, query: str) -> Dict[str, Any]:
        print("📄 Diagnosis: Processing")
        
        prompt = f"""
        Provide a best guess diagnosis for the list of symptoms based on the provided context. If the answer is not in the context, say "I don't have enough information to give a diagnosis."

        Context:
        {context}

        Symptoms: {query}

        Answer:
        """

        # Get structured information from Ollama
        diagnosis = self._query_ollama(ollama_model, prompt)

        return {
            "diagnosis": diagnosis,
            "status": "completed"
        }
