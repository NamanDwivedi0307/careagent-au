from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated
import operator
import json
from datetime import datetime

from src.agents.intake_agent import IntakeAgent
from src.memory.vector_store import ResidentMemory


# ── State definition ─────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state passed between all agents in the graph."""
    user_input: str
    resident_id: str
    resident_info: str
    intent: str
    care_plan: str
    risk_assessment: str
    medication_notes: str
    family_update: str
    compliance_check: str
    final_response: str
    messages: Annotated[list, operator.add]
    timestamp: str


# ── Orchestrator ──────────────────────────────────────────────
class CareOrchestrator:
    """
    Master orchestrator for CareAgent-AU.
    Uses LangGraph to route tasks between 4 specialist agents
    based on intent classification of the care worker's request.
    """

    INTENTS = {
        "new_intake": "Register a new resident and generate care plan",
        "medication_check": "Check medications, interactions or schedule",
        "family_update": "Generate family progress update or summary",
        "compliance_check": "Check NDIS or aged care compliance requirements",
        "general_query": "General question about a resident or care practice"
    }

    def __init__(self):
        print("Initialising CareAgent-AU Orchestrator...")
        self.llm = OllamaLLM(model="mistral", temperature=0.2)
        self.intake_agent = IntakeAgent()
        self.memory = ResidentMemory()
        self.graph = self._build_graph()
        print("Orchestrator ready — 4 agents online")

    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify the care worker's intent to route to correct agent."""
        prompt = PromptTemplate(
            input_variables=["user_input", "intents"],
            template="""You are a care coordination router.
Classify the following request into exactly one intent.

REQUEST: {user_input}

AVAILABLE INTENTS:
{intents}

Reply with ONLY the intent key, nothing else.
Example: new_intake"""
        )

        intent_list = "\n".join([f"- {k}: {v}" for k, v in self.INTENTS.items()])
        chain = prompt | self.llm | StrOutputParser()
        intent = chain.invoke({
            "user_input": state["user_input"],
            "intents": intent_list
        }).strip().lower()

        if intent not in self.INTENTS:
            intent = "general_query"

        print(f"Intent classified: {intent}")
        state["intent"] = intent
        state["messages"].append(f"[Router] Intent: {intent}")
        return state

    def _intake_node(self, state: AgentState) -> AgentState:
        """Run intake agent for new resident registration."""
        print("Running intake agent...")
        result = self.intake_agent.process_resident(state["resident_info"])

        state["care_plan"] = result["care_plan"]
        state["risk_assessment"] = result["risk_assessment"]

        if state.get("resident_id"):
            self.memory.save_resident(state["resident_id"], result)

        state["final_response"] = f"""
✅ INTAKE COMPLETE — {state.get('resident_id', 'New Resident')}

CARE PLAN:
{result['care_plan']}

RISK ASSESSMENT:
{result['risk_assessment']}
        """.strip()

        state["messages"].append("[IntakeAgent] Care plan generated")
        return state

    def _medication_node(self, state: AgentState) -> AgentState:
        """Check medications for a resident."""
        print("Running medication check...")

        # Retrieve resident from memory if ID provided
        context = ""
        if state.get("resident_id"):
            resident_data = self.memory.get_resident(state["resident_id"])
            if resident_data:
                context = resident_data.get("document", "")

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are an Australian aged care medication specialist.

RESIDENT CONTEXT:
{context}

QUERY: {query}

Provide a clear medication assessment covering:
1. CURRENT MEDICATIONS — list what is known
2. INTERACTION RISKS — any known interactions
3. SCHEDULE RECOMMENDATION — when to administer
4. MONITORING REQUIRED — what to watch for
5. GP REFERRAL NEEDED — yes or no and why

Use Australian medication guidelines and be specific."""
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": state["user_input"],
            "context": context
        })

        state["medication_notes"] = result
        state["final_response"] = f"💊 MEDICATION CHECK\n\n{result}"
        state["messages"].append("[MedicationAgent] Medication check complete")
        return state

    def _family_update_node(self, state: AgentState) -> AgentState:
        """Generate a family progress update."""
        print("Generating family update...")

        context = ""
        if state.get("resident_id"):
            resident_data = self.memory.get_resident(state["resident_id"])
            if resident_data:
                context = resident_data.get("document", "")

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a compassionate Australian aged care coordinator
writing a family update letter.

RESIDENT RECORDS:
{context}

REQUEST: {query}

Write a warm, professional family update that includes:
1. GENERAL WELLBEING — how the resident is doing overall
2. CARE HIGHLIGHTS — positive moments and achievements this week
3. HEALTH UPDATES — any relevant health changes (general terms only)
4. UPCOMING ACTIVITIES — what to look forward to
5. FAMILY VISIT SUGGESTIONS — best times and ways to engage

Write in a warm, reassuring tone. Use first name only for privacy.
Sign off as 'Your Care Team'."""
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "context": context,
            "query": state["user_input"]
        })

        state["family_update"] = result
        state["final_response"] = f"👨‍👩‍👧 FAMILY UPDATE\n\n{result}"
        state["messages"].append("[FamilyAgent] Family update generated")
        return state

    def _compliance_node(self, state: AgentState) -> AgentState:
        """Check NDIS and aged care compliance."""
        print("Running compliance check...")

        prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are an Australian NDIS and Aged Care compliance expert.

QUERY: {query}

Provide compliance guidance covering:
1. RELEVANT STANDARDS — which Aged Care Quality Standards apply
2. NDIS REQUIREMENTS — relevant NDIS practice standards
3. DOCUMENTATION NEEDED — what records must be kept
4. RISK OF NON-COMPLIANCE — potential consequences
5. RECOMMENDED ACTION — specific next steps

Reference the Aged Care Act 1997 and NDIS Act 2013 where relevant.
Be specific and practical for Australian care providers."""
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["user_input"]})

        state["compliance_check"] = result
        state["final_response"] = f"✅ COMPLIANCE CHECK\n\n{result}"
        state["messages"].append("[ComplianceAgent] Compliance check complete")
        return state

    def _general_query_node(self, state: AgentState) -> AgentState:
        """Handle general care queries."""
        print("Handling general query...")

        context = ""
        if state.get("resident_id"):
            similar = self.memory.search_similar(state["user_input"], n_results=2)
            if similar:
                context = similar[0]["document"]

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are an expert Australian aged care coordinator.

CONTEXT:
{context}

QUERY: {query}

Provide a helpful, practical response following Australian aged care
best practices. Be specific and actionable."""
        )

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": state["user_input"],
            "context": context
        })

        state["final_response"] = f"💬 CARE ASSISTANT\n\n{result}"
        state["messages"].append("[GeneralAgent] Query answered")
        return state

    def _route_intent(self, state: AgentState) -> str:
        """Route to correct agent based on classified intent."""
        return state["intent"]

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph multi-agent workflow."""
        graph = StateGraph(AgentState)

        # Add all nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("new_intake", self._intake_node)
        graph.add_node("medication_check", self._medication_node)
        graph.add_node("family_update", self._family_update_node)
        graph.add_node("compliance_check", self._compliance_node)
        graph.add_node("general_query", self._general_query_node)

        # Entry point
        graph.set_entry_point("classify_intent")

        # Conditional routing based on intent
        graph.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "new_intake": "new_intake",
                "medication_check": "medication_check",
                "family_update": "family_update",
                "compliance_check": "compliance_check",
                "general_query": "general_query"
            }
        )

        # All agents end after completing their task
        graph.add_edge("new_intake", END)
        graph.add_edge("medication_check", END)
        graph.add_edge("family_update", END)
        graph.add_edge("compliance_check", END)
        graph.add_edge("general_query", END)

        return graph.compile()

    def run(self, user_input: str, resident_id: str = "",
            resident_info: str = "") -> dict:
        """
        Run the orchestrator with a care worker's request.

        Args:
            user_input: The care worker's query or request
            resident_id: Optional resident ID for memory retrieval
            resident_info: Optional raw resident info for new intake

        Returns:
            dict with final_response and full state
        """
        print(f"\n{'='*60}")
        print("CareAgent-AU — Processing request")
        print(f"Input: {user_input[:80]}...")
        print(f"{'='*60}")

        initial_state = AgentState(
            user_input=user_input,
            resident_id=resident_id,
            resident_info=resident_info,
            intent="",
            care_plan="",
            risk_assessment="",
            medication_notes="",
            family_update="",
            compliance_check="",
            final_response="",
            messages=[],
            timestamp=datetime.now().isoformat()
        )

        final_state = self.graph.invoke(initial_state)

        print(f"\n{'='*60}")
        print("RESPONSE:")
        print(final_state["final_response"])
        print(f"{'='*60}\n")

        return final_state
