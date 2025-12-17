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

### 2. 📖 Fiction Analyzer & Story Bible
An intelligent "Editor Agent" that reads your manuscript in real-time to build a living Story Bible.

- **Dynamic Knowledge Base**: Automatically extracts and tracks:
    - **Characters**: Names, aliases, physical appearances, personality traits, and relationships.
    - **Plot Timeline**: Identifying key narrative beats and story progression.
    - **Locations**: Tracking settings and world-building details.
- **Smart Deduplication**: Uses fuzzy matching and logic to merge duplicate entities (e.g., realizing "The Captain" and "Captain Thorne" are the same person).
- **Tool-Use Architecture**: Powered by **GPT-4o**, the agent autonomously decides when to update the database vs. when to just read.

### 3. ✍️ Grammar & Style Editor
A specialized copy-editing assistant that goes beyond simple spell-checking.

- **Nuanced Analysis**: categorizes issues into **Critical** (Grammar), **Clarity** (Ambiguity), and **Style** (Suggestions).
- **Dashboard Metrics**: View a cleanliness score and track error frequency over time.
- **Historical View**: Compare current drafts against previous analysis runs.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Image Generation**: Google GenAI SDK (Gemini 3 Pro)
- **Natural Language Processing**: OpenAI API (GPT-4o) for logic, NLTK for tokenization.
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

- **Cover Generator Tab**: Configure your book size, genre, and design preferences. Click "Generate" to see the 3D preview.
- **Fiction Analyzer Tab**: Paste chapters or scenes into the text area to have them analyzed by the AI Editor.
