from typing import Dict, Any
from .base_agent import BaseAgent


class PrescriptionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Prescription",
            instructions="""Prescribe a drug regimen based on the given diagnosis.
            Use the provided context to help formulate an answer.
            Provide output in a clear, structured format."""
        )

    async def run(self, ollama_model: str, context: str, query: str) -> Dict[str, Any]:
        print("🔍 Prescription: Processing")
        
        prompt = f"""
        Provide a prescription drug regimen for the given diagnosis. If the answer is not in the context, say "I don't have enough information to give a diagnosis."

        Context:
        {context}

        Diagnosis: {query}

        Answer:
        """

        # Get structured information from Ollama
        prescription = self._query_ollama(ollama_model, prompt)

        return {
            "prescription": prescription,
            "status": "completed"
        }
