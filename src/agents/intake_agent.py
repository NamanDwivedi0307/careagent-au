from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import json


class IntakeAgent:
    """
    Intake Agent — CareAgent-AU
    Takes raw resident information and generates a structured
    care plan following Australian aged care standards.
    """

    def __init__(self):
        print("Initialising Intake Agent...")
        self.llm = OllamaLLM(model="mistral", temperature=0.3)
        self.care_plan_prompt = self._build_care_plan_prompt()
        self.risk_assessment_prompt = self._build_risk_prompt()
        print("Intake Agent ready")

    def _build_care_plan_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["resident_info"],
            template="""You are a professional Australian aged care coordinator.
Based on the resident information below, generate a structured care plan.
Follow Australian Aged Care Quality Standards and NDIS guidelines.

RESIDENT INFORMATION:
{resident_info}

Generate a care plan with these exact sections:
1. RESIDENT SUMMARY — name, age, key conditions
2. IMMEDIATE CARE NEEDS — top 3 urgent needs
3. DAILY CARE SCHEDULE — morning, afternoon, evening tasks
4. MEDICATION NOTES — any medications mentioned, frequency
5. MOBILITY AND SAFETY — fall risk and mobility aids needed
6. SOCIAL AND EMOTIONAL — wellbeing and family contact needs
7. REVIEW DATE — recommended next assessment date

Be specific, practical, and compassionate. Use Australian terminology.
Format each section clearly with the section header in capitals."""
        )

    def _build_risk_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["resident_info", "care_plan"],
            template="""You are an Australian aged care risk assessor.

RESIDENT INFORMATION:
{resident_info}

CARE PLAN:
{care_plan}

Assess the following risks and rate each LOW / MEDIUM / HIGH:
1. FALL RISK
2. MEDICATION RISK
3. SOCIAL ISOLATION RISK
4. COGNITIVE DECLINE RISK
5. NUTRITION RISK

For each risk, provide:
- Rating: LOW / MEDIUM / HIGH
- Reason: one sentence explanation
- Action: one specific intervention

Keep responses concise and clinically appropriate for Australian standards."""
        )

    def generate_care_plan(self, resident_info: str) -> str:
        """Generate a full care plan from resident information."""
        print("Generating care plan...")
        chain = self.care_plan_prompt | self.llm | StrOutputParser()
        care_plan = chain.invoke({"resident_info": resident_info})
        return care_plan.strip()

    def assess_risks(self, resident_info: str, care_plan: str) -> str:
        """Run risk assessment on the resident."""
        print("Running risk assessment...")
        chain = self.risk_assessment_prompt | self.llm | StrOutputParser()
        risks = chain.invoke({"resident_info": resident_info, "care_plan": care_plan})
        return risks.strip()

    def process_resident(self, resident_info: str) -> dict:
        """
        Full intake pipeline:
        1. Generate care plan
        2. Run risk assessment
        3. Return structured output
        """
        print(f"\n{'='*50}")
        print("INTAKE AGENT — Processing new resident")
        print(f"{'='*50}")

        care_plan = self.generate_care_plan(resident_info)
        risks = self.assess_risks(resident_info, care_plan)

        output = {
            "timestamp": datetime.now().isoformat(),
            "resident_info": resident_info,
            "care_plan": care_plan,
            "risk_assessment": risks,
            "agent": "intake_agent",
            "status": "completed"
        }

        print("\n✓ Care plan generated")
        print("✓ Risk assessment completed")
        print(f"{'='*50}\n")

        return output
