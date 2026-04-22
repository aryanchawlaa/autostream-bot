# AutoStream Lead Agent

A conversational AI agent that handles product queries and captures qualified leads for AutoStream, a video editing SaaS for content creators.

---

## Setup

### Requirements
- Python 3.9+
- Groq API key — get one free at https://console.groq.com

### Steps

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd autostream-lead-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
# Create a .env file and add:
GROQ_API_KEY=your_key_here

# 5. Run
python main.py
```

---

## How it works

The agent handles three types of user intent:
- **Greeting** — casual openers
- **Inquiry** — pricing and feature questions answered via RAG from products.json
- **High intent** — triggers lead collection (name → email → platform) before calling mock_lead_capture()

Sample conversation:
```
You: hi what are your plans?
Agent: We have two plans — Basic at $29/month (10 videos, 720p) and Pro at $79/month (unlimited, 4K, AI captions).

You: sounds good I want to try the pro plan for my YouTube channel
Agent: Great choice! Could I get your name?

You: Rahul Sharma
Agent: Thanks Rahul! What's your email?

You: rahul@gmail.com
Agent: Which platform do you create on?

You: YouTube
Agent: You're all set! Our team will reach out to rahul@gmail.com soon.

[console] Lead captured successfully: Rahul Sharma, rahul@gmail.com, YouTube
```

---

## Architecture (~200 words)

**Why LangGraph?**
LangGraph was chosen over AutoGen because it gives full control over state transitions through a typed `StateGraph`. For a lead-capture workflow, predictable step-by-step control matters more than autonomous multi-agent behavior. LangGraph makes it easy to enforce rules like "don't fire the tool until all three fields are collected."

**State Management**
The `AgentState` TypedDict carries the full conversation across turns:
- `messages` — complete history using LangGraph's `add_messages` reducer
- `intent` — classified each turn as greeting, inquiry, or high_intent
- `lead_name`, `lead_email`, `lead_platform` — filled one at a time via slot-filling
- `collecting_lead` and `lead_captured` — flags that control flow and prevent duplicate tool calls

**RAG**
Product data lives in `products.json` and is loaded into the system prompt at startup. This gives the LLM grounded, factual answers. In production this would use a vector database like Chroma with embedding-based retrieval for larger knowledge bases.

**LLM**
Uses `llama3-8b-8192` via Groq for fast, free inference.

---

## WhatsApp Integration via Webhooks

To deploy on WhatsApp Business API:

1. Register a webhook URL (e.g. `https://yourdomain.com/webhook`) in Meta Developer Console
2. On `GET /webhook` — return `hub.challenge` to verify
3. On `POST /webhook` — parse the incoming message and sender phone number
4. Use phone number as session key in Redis to load/save `AgentState` between turns
5. After running the agent, send the reply via `POST graph.facebook.com/v18.0/{PHONE_ID}/messages`

This approach keeps the agent stateful across WhatsApp's stateless webhook model using Redis as the session store.

---

## Files

```
autostream-lead-agent/
├── main.py          # Agent logic (LangGraph + Groq)
├── products.json    # Knowledge base (pricing, policies, FAQs)
├── requirements.txt
└── README.md
```
