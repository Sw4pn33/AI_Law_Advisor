<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Balance%20Scale.png" width="120"/>

# AI-Powered Legal Rights Awareness Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6F00?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Hybrid RAG + Legal NER System for Accessible Legal Information**

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500"/>

---

*A chatbot that uses **Hybrid Retrieval-Augmented Generation (RAG)**, **Legal Named Entity Recognition (NER)**, and **LLMs** to explain legal rights in plain language, retrieve accurate legal sections, generate formal legal notices, and provide referrals to legal aid organizations.*

> **Disclaimer:** This system provides **legal information**, not legal advice. Consult a qualified lawyer for your specific situation.

</div>

---

## <img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="25"> Features

| Feature | Description |
|---------|------------|
| **Hybrid Retrieval** | BM25 keyword search + ChromaDB vector semantic search with Reciprocal Rank Fusion (RRF) |
| **Legal NER** | Extracts statutes, sections, amounts, penalties, and courts from queries and responses |
| **Plain Language** | LLM translates complex legal jargon to 8th-grade reading level |
| **PDF Generator** | 6 templates — eviction, deposit, rent, consumer grievance, defective product, refund |
| **Dual LLM** | Google Gemini or Anthropic Claude, auto-detected from environment |
| **Legal Aid** | NALSA, e-Daakhil, DLSA, Tele-Law, Consumer Helpline, SHRC referrals |
| **Multilingual** | English and Hindi support |

---

## <img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="25"> Legal Domains (MVP)

```
 Tenant Rights       Consumer Rights       General Laws
  Rent disputes        Defective products     Cyber crime
  Security deposits    Refunds                Employment
  Eviction rules       Complaints             Traffic & family law
```

---

## <img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="25"> Tech Stack

<div align="center">

| Layer | Technology |
|:-----:|:----------:|
| **Frontend** | React 19, Tailwind CSS 3, React Router, React Markdown, Axios |
| **Backend** | FastAPI, Uvicorn |
| **RAG** | BM25 (rank-bm25), ChromaDB, Sentence Transformers |
| **NER** | spaCy (en_core_web_sm) + domain-specific regex |
| **LLM** | Google Generative AI / Anthropic Claude |
| **PDF** | ReportLab + Jinja2 |
| **Database** | PostgreSQL 18 (primary) / SQLite (fallback) |

</div>

---

## <img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" width="35"> Installation & Setup

### Prerequisites

```
 Python 3.10+     Node.js 18+     npm
```

### 1. Clone the Repository

```bash
git clone https://github.com/adyaomnkar/AI_Law_Advisor.git
cd AI_Law_Advisor
```

### 2. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# Add ONE of these (auto-detected):
GEMINI_API_KEY=your-gemini-api-key
# OR
ANTHROPIC_API_KEY=your-anthropic-api-key

# PostgreSQL (optional - falls back to SQLite if not set)
PG_HOST=localhost
PG_PORT=5432
PG_DB=law_advisor
PG_USER=postgres
PG_PASS=your_password
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Install & Start Backend

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```

> Backend runs on `http://localhost:8000`
> First startup downloads a ~79MB embedding model for ChromaDB (cached after that).
> Console shows `[DB] Using PostgreSQL` or `[DB] Using SQLite` depending on your config.

### 4. Install & Start Frontend

```bash
cd frontend
npm install
npm start
```

> Frontend runs on `http://localhost:3000`

---

## <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="40"> How It Works

```mermaid
graph LR
    A[User Query] --> B[Legal NER]
    B --> C[BM25 Search]
    B --> D[Vector Search]
    C --> E[RRF Fusion]
    D --> E
    E --> F[LLM Response]
    F --> G[React UI]
    G --> H[PDF Generator]

    style A fill:#4F46E5,color:#fff,stroke:none
    style B fill:#7C3AED,color:#fff,stroke:none
    style C fill:#2563EB,color:#fff,stroke:none
    style D fill:#2563EB,color:#fff,stroke:none
    style E fill:#D97706,color:#fff,stroke:none
    style F fill:#059669,color:#fff,stroke:none
    style G fill:#0891B2,color:#fff,stroke:none
    style H fill:#DC2626,color:#fff,stroke:none
```

1. User asks a legal question
2. System extracts legal entities (sections, statutes, amounts)
3. Hybrid search finds relevant provisions using BM25 + ChromaDB
4. Results are fused using Reciprocal Rank Fusion
5. LLM generates a plain-language response with citations
6. User can generate a formal legal notice PDF

---

## <img src="https://user-images.githubusercontent.com/74038190/212284136-03988914-d899-44b4-b1d9-4c09e4c6e6d2.gif" width="25"> API Endpoints

| Method | Endpoint | Description |
|:------:|----------|-------------|
| `GET` | `/` | Health check, LLM provider status |
| `POST` | `/api/chat` | Send query, get legal response with entities & sources |
| `POST` | `/api/search` | Search legal documents |
| `POST` | `/api/generate-pdf` | Generate legal notice PDF |
| `GET` | `/api/legal-aid` | Get legal aid services list |
| `GET` | `/api/sessions` | Get chat history |

---

## <img src="https://user-images.githubusercontent.com/74038190/216120981-b9507c36-0e04-4469-8e27-c99271b45ba5.png" width="25"> Demo

**Query:**
> "I bought a fridge and it broke the next day. The shop refuses to refund my 20,000 rupees."

**System Response:**
- Identifies: Product (Fridge), Amount (Rs. 20,000), Issue (Defective Product)
- Retrieves: Consumer Protection Act sections
- Explains rights in plain language with actionable steps
- Option to generate a Refund Request legal notice PDF

---

## Project Structure

```
AI_Law_Advisor/
├── backend/
│   ├── main.py                 # FastAPI app, routes
│   ├── config.py               # Environment config, LLM detection
│   ├── llm_service.py          # Dual LLM service (Gemini/Claude)
│   ├── pdf_generator.py        # Legal notice PDF generation
│   ├── requirements.txt
│   ├── database/
│   │   └── db.py               # PostgreSQL / SQLite
│   ├── rag_pipeline/
│   │   └── hybrid_search.py    # BM25 + ChromaDB + RRF
│   ├── ner/
│   │   └── legal_ner.py        # Legal entity extraction
│   └── templates/
│       └── legal_notice.html   # Jinja2 HTML template
│
├── frontend/
│   └── src/
│       ├── App.js
│       ├── context/ChatContext.js
│       ├── services/api.js
│       ├── pages/
│       │   ├── ChatPage.js
│       │   ├── LegalAidPage.js
│       │   └── AboutPage.js
│       └── components/
│           ├── Navbar.js
│           ├── ChatInput.js
│           ├── ChatMessage.js
│           ├── WelcomeScreen.js
│           ├── PDFGenerator.js
│           ├── DomainSelector.js
│           ├── EntityBadge.js
│           └── Disclaimer.js
│
├── data/
│   ├── tenant_rights.txt
│   ├── consumer_protection.txt
│   └── general_laws.txt
│
├── .env.example
└── .gitignore
```

---

<div align="center">

## License

MIT License

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="120"/>

*Built for accessible justice*

</div>
