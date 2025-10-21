Perfect — here’s a professional, complete, and GitHub-ready README for your project 🧠📱

📱 Phone Recommendation Agent

An AI-powered multi-component system that analyzes phone specifications, scrapes Reddit reviews, and uses an LLM reasoning layer to generate smart recommendations.
Built with FastAPI, MCP (Multi-Component Protocol) tools, and a Streamlit frontend — this project demonstrates agentic reasoning, retrieval, and recommendation fusion.

🚀 Features

✅ LLM-driven reasoning:
Uses a large language model (GPT-based) to interpret user queries, evaluate device specs, and summarize opinions.

✅ MCP Tools Integration:

Phone Specs Tool – fetches detailed phone specifications (brand, model, chipset, etc.)

Reddit Reviews Tool – retrieves and summarizes recent real-world feedback from Reddit.

Recommendation Tool – synthesizes structured reasoning into a concise, human-like verdict.

✅ Streamlit Interface:
Interactive app that wraps all components, allowing users to:

Input any phone name (e.g., “Samsung A24”)

Fetch top specs and Reddit feedback

Get an LLM-based final recommendation

✅ FastAPI Backend:
Handles the orchestration between MCP client, LLM agent, and external APIs.

🧩 Architecture
Phone-recommendation-Agent/
│
├── Backend/
│   ├── mcp_server.py       # MCP Server exposing phone tools
│   ├── mcp_client.py       # Client interface for MCP server
│   ├── llm_agent.py        # LLM reasoning engine
│   ├── test_mcp.py         # MCP tool testing script
│
├── streamlit_app.py        # Unified frontend for interactive use
├── requirements.txt        # Python dependencies
└── README.md

⚙️ Installation
1️⃣ Clone the repo
git clone https://github.com/<your-username>/Phone-recommendation-Agent.git
cd Phone-recommendation-Agent

2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run streamlit_app.py

🧠 How It Works

User Input: Enter a phone name in the Streamlit app.

Specs Tool: The backend fetches and ranks top matches via the MCP spec tool.

Reddit Tool: Real user opinions are pulled and summarized.

LLM Agent: Combines structured specs and user reviews to form a final recommendation.

Display: Streamlit renders everything (Specs, Reviews, Verdict).

🧪 Testing MCP Tools

You can test the backend tools individually with:

python Backend/test_mcp.py


It prints top matches and reviews in the console for debugging.

💡 Example Output

Input: Samsung A24

Specs (Top Match):

Model: Galaxy A24
Chipset: MediaTek Helio G99
Released: 2023-04-01


Reddit Reviews (Summarized):

“Battery life is decent, but performance lags under heavy load.”
“Not worth it if you game or multitask heavily.”

LLM Recommendation:

“The Galaxy A24 is fine for casual users but not ideal for gamers or power users. Consider upgrading to the A34 or S20 FE for smoother experience.”

🧠 Tech Stack
Component	Technology
Frontend	Streamlit
Backend	FastAPI
Reasoning	OpenAI GPT / Local LLM
Tools	MCP (Multi-Component Protocol)
Data Sources	Reddit API, Device Specs DB
Language	Python 3.10+
🧩 Future Work

🧠 Add RAG (Retrieval-Augmented Generation) for deeper device comparison

💬 Implement conversational recommendation chat mode

📈 Integrate user feedback to improve recommendations

⚙️ Add Docker support for portable deployment

👨‍💻 Author

Ahmed Hammadi
Agentic AI Developer & MLOps Engineer
📧 [your-email@example.com
]
🌐 [your-portfolio-link.com]

🪪 License

This project is licensed under the MIT License — see the LICENSE file for details.