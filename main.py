import json
import os
import re
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


def load_data(path: str = "products.json") -> str:
    with open(path, "r") as f:
        kb = json.load(f)

    text = f"""
COMPANY: {kb['company']}
DESCRIPTION: {kb['description']}

PRICING PLANS:
- Basic Plan: {kb['plans']['basic']['price']}
  Features: {', '.join(kb['plans']['basic']['features'])}

- Pro Plan: {kb['plans']['pro']['price']}
  Features: {', '.join(kb['plans']['pro']['features'])}

POLICIES:
- Refund Policy: {kb['policies']['refund']}
- Support: {kb['policies']['support']}
- Trial: {kb['policies']['trial']}
- Cancellation: {kb['policies']['cancellation']}

FAQs:
"""
    for faq in kb['faqs']:
        text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return text.strip()


PRODUCT_DATA = load_data()


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n{'='*50}")
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = f"""You are an AI sales assistant for AutoStream, a SaaS platform that provides automated video editing tools for content creators.

Your job is to:
1. Greet users warmly
2. Answer questions about AutoStream's pricing and features using ONLY the knowledge base below
3. Detect when a user is ready to sign up (high intent)
4. Collect their Name, Email, and Creator Platform (YouTube, Instagram, TikTok, etc.) ONE AT A TIME before capturing the lead
5. Never ask for all details at once — ask one question at a time

KNOWLEDGE BASE:
{PRODUCT_DATA}

INTENT CLASSIFICATION:
- "greeting": Simple hellos, how are you, etc.
- "inquiry": Questions about pricing, features, policies
- "high_intent": User explicitly wants to sign up, try, or purchase a plan

LEAD COLLECTION RULES:
- Only start collecting lead info when you detect HIGH INTENT
- Ask for Name first, then Email, then Creator Platform
- After collecting all three, confirm the lead has been captured

Always be friendly, concise, and helpful.
"""


def detect_intent(state: AgentState) -> AgentState:
    if not state["messages"]:
        return state

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        return state

    user_text = last_msg.content.lower()

    high_intent_keywords = [
        "sign up", "signup", "subscribe", "buy", "purchase", "get started",
        "try", "want to", "i'm interested", "let's go", "sounds good",
        "i want", "take the", "enroll", "register", "start my", "upgrade"
    ]

    if any(kw in user_text for kw in high_intent_keywords):
        intent = "high_intent"
    elif any(w in user_text for w in ["hi", "hello", "hey", "howdy", "good morning", "good evening"]):
        if not any(w in user_text for w in ["price", "plan", "feature", "cost", "how much", "support"]):
            intent = "greeting"
        else:
            intent = "inquiry"
    else:
        intent = "inquiry"

    return {**state, "intent": intent}


def extract_email(text: str) -> Optional[str]:
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_platform(text: str) -> Optional[str]:
    platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook", "twitch", "linkedin", "snapchat"]
    for p in platforms:
        if p in text.lower():
            return p.capitalize()
    return None


def agent_node(state: AgentState) -> AgentState:
    new_state = dict(state)

    if state.get("collecting_lead") and not state.get("lead_captured"):
        last_human = state["messages"][-1].content if state["messages"] else ""

        if not state.get("lead_name"):
            new_state["lead_name"] = last_human.strip()
            response_text = f"Thanks, {new_state['lead_name']}! What's your email address?"

        elif not state.get("lead_email"):
            email = extract_email(last_human) or last_human.strip()
            new_state["lead_email"] = email
            response_text = "Got it! Which creator platform do you mainly use? (YouTube, Instagram, TikTok, etc.)"

        elif not state.get("lead_platform"):
            platform = extract_platform(last_human) or last_human.strip()
            new_state["lead_platform"] = platform

            mock_lead_capture(
                new_state["lead_name"],
                new_state["lead_email"],
                new_state["lead_platform"]
            )
            new_state["lead_captured"] = True
            new_state["collecting_lead"] = False
            response_text = (
                f"You're all set, {new_state['lead_name']}! "
                f"Our team will reach out to you at {new_state['lead_email']} soon. "
                f"Can't wait to help you grow on {new_state['lead_platform']} with AutoStream!"
            )
        else:
            response_text = "Looks like we already have your info! Anything else I can help with?"

        new_state["messages"] = state["messages"] + [AIMessage(content=response_text)]
        return new_state

    intent = state.get("intent", "inquiry")
    intent_note = ""
    if intent == "high_intent":
        intent_note = "\n\n[NOTE: This user has HIGH INTENT to sign up. After responding, ask for their name to begin lead collection.]"

    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT + intent_note)] + list(state["messages"])
    response = llm.invoke(messages_for_llm)
    new_state["messages"] = state["messages"] + [AIMessage(content=response.content)]

    if intent == "high_intent" and not state.get("lead_captured"):
        new_state["collecting_lead"] = True
        updated = response.content + "\n\nTo get you started, could I get your name?"
        new_state["messages"] = state["messages"] + [AIMessage(content=updated)]

    return new_state


def should_end(state: AgentState) -> str:
    if state.get("lead_captured"):
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "agent")
    graph.add_conditional_edges("agent", should_end, {"end": END, "continue": END})
    return graph.compile()


def run():
    print("\n" + "="*55)
    print("  AutoStream Sales Assistant")
    print("  Type 'exit' to quit")
    print("="*55 + "\n")

    app = build_graph()

    state: AgentState = {
        "messages": [],
        "intent": "greeting",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    opening = (
        "Hey! Welcome to AutoStream. I can help with pricing, features, or getting you signed up.\n"
        "What brings you here today?"
    )
    print(f"Agent: {opening}\n")
    state["messages"].append(AIMessage(content=opening))

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAgent: Thanks for stopping by! Have a good one!\n")
            break

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state = result

        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            print(f"\nAgent: {ai_messages[-1].content}\n")

        if state.get("lead_captured"):
            print("-" * 55)
            print("Lead successfully captured. Ending session.")
            print("-" * 55)
            break


if __name__ == "__main__":
    run()
