from typing import Dict, Any
from .base_agent import BaseAgent
from .diagnosis_agent import DiagnosisAgent
from .prescription_agent import PrescriptionAgent


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            instructions="""Coordinate the workflow and delegate tasks to specialized agents.
            Ensure proper flow of information between diagnosis and prescription phases.
            Maintain context and aggregate results from each stage.""",
        )
        self._setup_agents()

    def _setup_agents(self):
        """Initialize all specialized agents"""
        self.diagnosis = DiagnosisAgent()
        self.prescription = PrescriptionAgent()

    async def run(self, messages: list) -> Dict[str, Any]:
        """Process a single message through the agent"""
        prompt = messages[-1]["content"]
        response = self._query_ollama(prompt)
        return self._parse_json_safely(response)

    async def process_symptoms(self, ollama_model: str, context: str, query: str) -> Dict[str, Any]:
        """Main workflow orchestrator for processing symptoms"""
        print("🎯 Orchestrator: Starting process")

        workflow_context = {
            "query": query,
            "status": "initiated",
            "current_stage": "diagnosis",
        }

        try:
            # Diagnosis
            diagnosis_data = await self.diagnosis.run(ollama_model, context, query)
            workflow_context.update(
                {"diagnosis_data": diagnosis_data, "current_stage": "prescription"}
            )

            # Prescription
            prescription_data = await self.prescription.run(ollama_model, context, diagnosis_data['diagnosis'])
            workflow_context.update(
                {"prescription_data": prescription_data, "status": "completed"}
            )

            return workflow_context

        except Exception as e:
            workflow_context.update({"status": "failed", "error": str(e)})
            raise
