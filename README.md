# Bellwether Studio

**Bellwether Studio** is an all-in-one creative suite for authors, editors, and independent publishers. It merges advanced AI specifically for narrative analysis with state-of-the-art visual generation tools, providing a seamless workflow from drafting to publishing.

## 🚀 Key Features

### 1. 🎨 AI Book Cover Generator
Generate professional, print-ready book covers with full wrap-around designs (Front, Spine, Back).

- **Powered by Gemini 3 Pro**: Utilizes `gemini-3-pro-image-preview` for high-fidelity, text-aware image generation.
- **Interactive 3D Preview**: Instantly visualize your design on a rotatable 3D book model directly in the browser.
- **Print-Ready Specifications**:
    - **Industry Standard Sizes**: Built-in presets for Mass Market, Trade Paperback, and Hardcover formats.
    - **Smart Spine Calculation**: Automatically computes spine width based on page count and paper type (Cream/White/Color).
    - **Full Bleed Support**: Generates dimensions with standard bleed and safe zones automatically included.
- **JSON Design Export**: Download full design specifications and prompts for professional designers or archival.

### 2. 📖 Developmental Edit (Fiction)
An intelligent "Editor Agent" that reads your manuscript in real-time to build a living Story Bible.

- **Dynamic Knowledge Base**: Automatically extracts and tracks:
    - **Characters**: Names, aliases, physical appearances, personality traits, and relationships.
    - **Plot Timeline**: Identifying key narrative beats and story progression.
    - **Locations**: Tracking settings and world-building details.
- **Smart Deduplication**: Uses fuzzy matching and logic to merge duplicate entities.
- **Tool-Use Architecture**: Powered by **GPT-4o**, the agent autonomously decides when to update the database vs. when to just read.

### 3. 🧠 Non-Fiction Analyzer
A dedicated pipeline for technical texts, essays, and research papers.

- **Concept Extraction**: Identifies key arguments, core concepts, and claims.
- **Metric Tracking**: Automatically pulls out statistics and data points.
- **Structure Analysis**: Maps out the logical flow of the argument.

### 4. ✍️ Grammar & Style Editor (Integrated)
A centralized grammar dashboard that analyzes text from both the Fiction and Non-Fiction pipelines.

- **Hybrid Analysis Engine**:
    - **Rule-Based**: Deterministic checks for tense consistency, modal verbs, and style rules (using NLTK & LanguageTool).
    - **Spell Check**: Robust, offline spell checking via `pyspellchecker`.
    - **LLM Context**: Uses **GPT-4o-mini** (Temperature 0) to catch subtle contextual errors (e.g., "affect" vs "effect", passive voice) without hallucinating.
- **Dashboard Metrics**: View a cleanliness score and track error frequency over time.
- **Annotated View**: See errors highlighted directly in your text with tooltips and suggested fixes.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Image Generation**: Google GenAI SDK (Gemini 3 Pro)
- **Natural Language Processing**: OpenAI API (GPT-4o) for logic, NLTK & Spacy for tokenization.
- **Grammar & Spelling**: `pyspellchecker`, `language-tool-python`, and custom deterministic rules.
- **Data Management**: Pydantic for strict schema validation.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- A Google Cloud API Key (with access to Gemini models)
- An OpenAI API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cover_image_generation
   ```

2. **Create and activate a virtual environment:**
   Using `uv` (recommended):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   Or using standard `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_ai_key
   OPENAI_API_KEY=your_openai_key
   ```

## 🖥️ Usage

Run the main application:
```bash
streamlit run app.py
```

- **Cover Generator**: Configure your book size, genre, and design preferences to generate print-ready covers with 3D previews.
- **Fiction Analyzer**: Paste chapters to extract character/plot details and run background grammar checks.
- **Non-Fiction Analyzer**: Analyze technical texts for concepts and claims.
- **Grammar & Style**: View the consolidated report of all grammar, spelling, and style issues found in your analyzed texts.
