import streamlit as st
import nltk
import json
import os
import io
import sys
from io import StringIO
from dotenv import load_dotenv
from cover_generator_ui import render_cover_generator

load_dotenv()





def chunk_text(text, chunk_size=200, overlap=50):
    """Chunks text into overlapping segments using NLTK tokenization."""
    # Ensure punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
             nltk.download('punkt', quiet=True)
             nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            st.warning(f"Could not download NLTK data: {e}. Fallback to basic splitting.")
            # Fallback to simple split if NLTK fails
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    words = nltk.word_tokenize(text)
    chunks = []
    
    # Handle empty or short text
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    step = chunk_size - overlap
    if step < 1: 
        step = 1
        
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
    return chunks

def render_story_so_far(agent, state):
    """Generates a short narrative summary of the current state."""
    if not state.plot_points:
        return
    
    st.subheader("Story Overview")
    
    # We use a cache key based on the number of plot points to avoid re-generating on every rerun
    # unless the state changed.
    state_hash = len(state.plot_points) + len(state.characters)
    if "summary_cache" not in st.session_state or st.session_state.summary_cache_key != state_hash:
        with st.spinner("Generating story summary..."):
            try:
                # Use the agent's client (OpenAI)
                prompt = (
                    "Based on the following plot points, write a 4-6 line narrative summary of the story so far. "
                    "Focus on the main stakes, decisions, and current situation.\n\n"
                    f"Plot Points: {[p.description for p in state.plot_points]}"
                )
                response = agent.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content
                st.session_state.summary_cache = summary
                st.session_state.summary_cache_key = state_hash
            except Exception as e:
                st.session_state.summary_cache = "Could not generate summary."
                print(f"Summary Error: {e}")

    st.markdown(f"*{st.session_state.summary_cache}*")
    st.divider()

def render_pretty_character(char):
    """Renders a single character in a clean, editor-friendly format."""
    # Header: Name + Aliases
    title = f"**{char.name}**"
    if char.aliases:
        title += f" <span style='color:gray; font-size:0.9em'>({', '.join(char.aliases)})</span>"
    st.markdown(title, unsafe_allow_html=True)
    
    # Description
    if char.appearance:
        st.markdown(f"{char.appearance}")

    # Traits Tags
    if char.traits:
        tags = [f"`{t}`" for t in char.traits]
        st.markdown(" ".join(tags))

    # Relationships
    if char.relationships:
        rel_text = [f"**{k}**: {v}" for k, v in char.relationships.items()]
        st.markdown(f"<small>{' | '.join(rel_text)}</small>", unsafe_allow_html=True)

    # Evidence (Expander)
    if char.evidence:
        with st.expander(f"Show Evidence ({len(char.evidence)})"):
            # Show top 3
            for i, e in enumerate(char.evidence[:3]):
                st.caption(f"\"{e}\"")
            if len(char.evidence) > 3:
                st.caption(f"...and {len(char.evidence)-3} more.")
    
    st.markdown("---")

def render_pretty_plot(plot_points):
    """Renders plot points as a timeline."""
    st.markdown("### Plot Timeline")
    for i, pp in enumerate(plot_points):
        # Format: 1. Description
        st.markdown(f"**{i+1}.** {pp.description}")
        if pp.evidence:
             with st.expander("Evidence"):
                 for e in pp.evidence:
                     st.caption(f"\"{e}\"")


def render_fiction_analyzer():
    st.header("Developmental Edit")
    st.write("Input text from your story (e.g., 2000 words) to build a knowledge base. The text will be automatically chunked (500 words, 100 overlap) and processed.")

    # Import here to avoid issues if file missing
    try:
        from agents.tool_use_agent import Agent, FictionState
        from agents.grammar_agent import GrammarAgent
    except Exception as e:
        st.error(f"Could not import backend agents. Error: {e}")
        return

    # Initialize Session State for Agent and Knowledge Base
    if "fiction_state" not in st.session_state:
        st.session_state.fiction_state = FictionState()
    
    if "grammar_reports" not in st.session_state:
        st.session_state.grammar_reports = []

    if "grammar_history" not in st.session_state:
        st.session_state.grammar_history = []  # List of {time, snippet, reports}
            
    # Check for API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OpenAI API Key not found in environment variables. Please add it to your .env file.")
        return
            
    agent = Agent()
    
    # Initialize Grammar Agent (might trigger large download on first run)
    if "grammar_agent_instance" not in st.session_state:
        with st.spinner("Initializing Grammar Engine (Downloading models on first run... this may take a minute)"):
            st.session_state.grammar_agent_instance = GrammarAgent()
    grammar_agent = st.session_state.grammar_agent_instance
    
    # Input Area
    chunk_input = st.text_area("Enter Story Text", height=300, placeholder="Paste a large segment of your story here (e.g., a full scene or chapter)...")
    
    if st.button("Process Text"):
        if chunk_input.strip():
            with st.spinner("Tokenizing and analyzing text..."):
                try:
                    # Chunk the input
                    chunks = chunk_text(chunk_input, chunk_size=500, overlap=100)
                    st.info(f"Text split into {len(chunks)} paragraph(s) for processing.")
                    
                    # Log buffer
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    result_buffer = StringIO()
                    sys.stdout = result_buffer
                    
                    # Clear previous grammar reports
                    st.session_state.grammar_reports = []

                    # Process each chunk
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        # Update progress
                        progress_bar.progress((i + 1) / len(chunks))
                        print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
                        
                        # 1. Knowledge Extraction
                        agent.process_chunk(st.session_state.fiction_state, chunk)
                        
                        # 2. Grammar Analysis
                        print(" >> Analyzing Grammar...")
                        report = grammar_agent.analyze_chunk(chunk)
                        if report.grammar_issues:
                            st.session_state.grammar_reports.append({
                                "chunk_index": i + 1,
                                "issues": report.grammar_issues,
                                "text": chunk
                            })

                    # Auto-Resolve Entities
                    print("--- Resolving Entities ---")
                    st.info("Auto-resolving entities...")
                    res_msg = agent.resolve_entities(st.session_state.fiction_state)
                    print(f"Resolution Result: {res_msg}")

                    sys.stdout = old_stdout
                    logs = result_buffer.getvalue()
                    
                    st.success(f"Analysis & Resolution Complete! ({res_msg})")
                    
                    # Store in History
                    from datetime import datetime
                    history_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "snippet": chunk_input[:60].replace("\n", " ") + "...",
                        "reports": st.session_state.grammar_reports, 
                        "stats": {
                            "high": sum(sum(1 for x in r['issues'] if x.severity == 'high') for r in st.session_state.grammar_reports),
                            "medium": sum(sum(1 for x in r['issues'] if x.severity == 'medium') for r in st.session_state.grammar_reports)
                        }
                    }
                    st.session_state.grammar_history.insert(0, history_entry) # Add to top

                    # Store logs in expader but closed
                    with st.expander("Debug Logs (Extraction Details)"):
                        st.code(logs)
                            
                except Exception as e:
                    st.error(f"Error processing text: {e}")
        else:
            st.warning("Please enter some text.")

    
    # --- PRESENTATION LAYER ---
    st.markdown("---")
    
    # 1. Story So Far
    render_story_so_far(agent, st.session_state.fiction_state)

    col1, col2 = st.columns([1, 1], gap="large")

    # 2. Characters Column
    with col1:
        st.header("Characters")
        if st.session_state.fiction_state.characters:
            for char in st.session_state.fiction_state.characters:
                render_pretty_character(char)
        else:
            st.info("No characters identified yet.")

    # 3. Plot & Locations Column
    with col2:
        # Plot
        if st.session_state.fiction_state.plot_points:
            render_pretty_plot(st.session_state.fiction_state.plot_points)
        else:
            st.markdown("### Plot Timeline")
            st.info("No plot points identified yet.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Locations
        st.subheader("Settings & Locations")
        if st.session_state.fiction_state.locations:
            for loc in st.session_state.fiction_state.locations:
                st.markdown(f"**📍 {loc.name}**")
                if loc.description:
                    st.caption(loc.description)
        else:
            st.info("No locations identified.")
        
    # Raw JSON View (Hidden)
    with st.expander("View Raw JSON State"):
        st.code(st.session_state.fiction_state.model_dump_json(indent=2), language="json")



def render_grammar_dashboard():
    st.header("Grammar & Style Editor")
    
    # Ensure history list exists
    if "grammar_history" not in st.session_state:
        st.session_state.grammar_history = []
    if "grammar_reports" not in st.session_state:
        st.session_state.grammar_reports = []

    # --- 1. HISTORY DROPDOWN ---
    if st.session_state.grammar_history:
        with st.expander("📜 Past Analysis History", expanded=False):
            # We use a selectbox to effectively "View" a past entry
            # Create labels
            options = {f"{e['timestamp']} - {e['snippet']} (Crit:{e['stats']['high']})": e for e in st.session_state.grammar_history}
            selected_label = st.selectbox("Select a previous run:", list(options.keys()))
            
            if selected_label:
                entry = options[selected_label]
                st.info(f"Viewing Historical Run from {entry['timestamp']}")
                reports_to_render = entry['reports']
            else:
                reports_to_render = st.session_state.grammar_reports
    else:
        reports_to_render = st.session_state.grammar_reports

    current_reports = reports_to_render

    if current_reports:
        # Flatten issues for stats
        all_issues = [issue for chunk in current_reports for issue in chunk['issues']]
        count_high = sum(1 for x in all_issues if x.severity == 'high')
        count_med = sum(1 for x in all_issues if x.severity == 'medium')
        count_low = sum(1 for x in all_issues if x.severity == 'low')
        
        # --- DASHBOARD METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Critical Errors", count_high, delta=None, delta_color="inverse")
        m2.metric("Clarity Warnings", count_med, delta=None, delta_color="normal")
        m3.metric("Suggestions", count_low, delta=None, delta_color="off")
        score = max(0, 100 - (count_high * 5) - (count_med * 2))
        m4.metric("Cleanliness Score", f"{score}/100")
        
        st.divider()
        
        # --- FILTERS ---
        with st.expander("Refine View (Filters)", expanded=True):
            f1, f2 = st.columns(2)
            severity_filter = f1.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium"])
            type_filter = f2.multiselect("Issue Type", ["spelling", "grammar", "clarity", "style"], default=["spelling", "grammar", "clarity", "style"])
        
        # --- RENDER CARDS ---
        for item in current_reports:
            # Filter Issues
            chunk_issues = [
                i for i in item['issues'] 
                if i.severity in severity_filter 
                and i.type in type_filter
            ]
            
            if not chunk_issues and (severity_filter or type_filter):
                continue # Skip empty chunks if filters match nothing

            st.markdown(f"### Paragraph {item['chunk_index']}")
            
            # Check for system/critical errors first
            system_errors = [i for i in item['issues'] if i.type == 'system']
            if system_errors:
                for sys_err in system_errors:
                    st.error(f"⚠️ {sys_err.issue}: {sys_err.explanation}")

            render_text = item.get("text", "")
            if render_text:
                # --- ANNOTATED VIEW ---
                sorted_issues = sorted(chunk_issues, key=lambda x: getattr(x, 'start_char', 0))
                
                html_content = ""
                last_idx = 0
                
                for issue in sorted_issues:
                    start = getattr(issue, 'start_char', 0)
                    end = getattr(issue, 'end_char', 0)
                    
                    # Safety check for indices
                    if start < last_idx: continue # Skip overlapping if logic failed
                    if end > len(render_text): end = len(render_text)
                    
                    # Text before
                    html_content += render_text[last_idx:start]
                    
                    # Highlight
                    severity = issue.severity
                    color = "#ffebee" if severity == "high" else "#fff3e0" if severity == "medium" else "#e3f2fd"
                    border = "#ef5350" if severity == "high" else "#ff9800" if severity == "medium" else "#2196f3"
                    
                    tooltip = f"[{issue.type.upper()}] {issue.issue}\n{issue.explanation}\nSuggestion: {issue.suggested_fix}"
                    snippet = render_text[start:end]
                    
                    html_content += f'<span style="background-color: {color}; border-bottom: 2px solid {border}; cursor: help;" title="{tooltip}">{snippet}</span>'
                    
                    last_idx = end
                
                html_content += render_text[last_idx:]
                
                st.markdown(f'<div style="font-family:serif; font-size:1.1em; line-height:1.6; padding:15px; background:white; border-radius:5px; border:1px solid #ddd;">{html_content}</div>', unsafe_allow_html=True)
                
                # Details Table
                with st.expander("Show Details & Fixes"):
                        for issue in sorted_issues:
                            st.markdown(f"**{issue.issue}** ({issue.severity}): {issue.explanation} -> `{issue.suggested_fix}`")
            else:
                # Fallback to Cards for old history
                for issue in chunk_issues:
                    st.info(f"{issue.issue}: {issue.explanation}")
    else:
        st.info("Run analysis in Fiction or Non-Fiction tabs to generate a Grammar Report.")


def main():
    st.set_page_config(page_title="Bellwether Studio", layout="wide")
    
    st.title("Bellwether Studio")

    # Initialize session state for dimensions if not present (needed for cover generator)
    if "trim_width" not in st.session_state:
        st.session_state.trim_width = 6.0
    if "trim_height" not in st.session_state:
        st.session_state.trim_height = 9.0

    tab1, tab2, tab3, tab4 = st.tabs(["Cover Generator", "Fiction Analyzer", "Non-Fiction Analyzer", "Grammar & Style"])
    
    with tab1:
        render_cover_generator()
    
    with tab2:
        render_fiction_analyzer()

    with tab3:
        render_non_fiction_analyzer()
        
    with tab4:
        render_grammar_dashboard()

def render_non_fiction_analyzer():
    st.header("Non-Fiction Knowledge Base Analyzer")
    st.write("Input technical texts, research papers, or essays. The system will extract concepts, claims, metrics, and processes.")

    # Import backend
    try:
        from agents.non_fiction_agent import Agent as NonFictionAgent, NonFictionState
        from agents.grammar_agent import GrammarAgent
    except Exception as e:
        st.error(f"Could not import Agents. Error: {e}")
        return

    # Initialize State
    if "non_fiction_state" not in st.session_state:
        st.session_state.non_fiction_state = NonFictionState()

    agent = NonFictionAgent()
    
    if "grammar_agent_instance" not in st.session_state:
        with st.spinner("Initializing Grammar Engine..."):
            st.session_state.grammar_agent_instance = GrammarAgent()
    grammar_agent = st.session_state.grammar_agent_instance
    
    if "grammar_reports" not in st.session_state:
        st.session_state.grammar_reports = []
    if "grammar_history" not in st.session_state:
        st.session_state.grammar_history = []

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Reset Knowledge Base"):
            st.session_state.non_fiction_state = NonFictionState()
            st.rerun()

    # Input
    chunk_input = st.text_area("Enter Non-Fiction Text", height=300, key="nf_input", placeholder="Paste article or paper content here...")

    if st.button("Analyze Knowledge"):
        if chunk_input.strip():
            with st.spinner("Extracting knowledge structure..."):
                try:
                    chunks = chunk_text(chunk_input, chunk_size=500, overlap=50)
                    st.info(f"Processing {len(chunks)} chunks...")
                    
                    progress_bar = st.progress(0)
                    
                    # Capture printed logs
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    result_buffer = StringIO()
                    sys.stdout = result_buffer

                    st.session_state.grammar_reports = []
                    for i, chunk in enumerate(chunks):
                        progress_bar.progress((i + 1) / len(chunks))
                        print(f"--- Processing Chunk {i+1} ---")
                        
                        # 1. Info Extraction
                        agent.process_chunk(st.session_state.non_fiction_state, chunk)
                        
                        # 2. Grammar
                        print(f"--- Analyzing Grammar for Chunk {i+1} ---")
                        report = grammar_agent.analyze_chunk(chunk)
                        if report.grammar_issues:
                            st.session_state.grammar_reports.append({
                                "chunk_index": i + 1,
                                "issues": report.grammar_issues,
                                "text": chunk
                            })
                    
                    # Auto-Resolve Entities
                    print("--- Resolving Entities ---")
                    st.info("Auto-resolving entities...")
                    res = agent.resolve_entities(st.session_state.non_fiction_state)
                    print(f"Resolution Result: {res}")
                    
                    sys.stdout = old_stdout
                    logs = result_buffer.getvalue()
                    
                    st.success(f"Extraction & Resolution Complete! ({res})")
                    with st.expander("Extraction Logs"):
                        st.code(logs)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")

    # --- VIEW & TOOLS ---
    st.divider()

    st.markdown("### 🧠 Current Knowledge Base")
    
    # Render Pretty View
    st.markdown(st.session_state.non_fiction_state.get_pretty_view())
    
    # Raw JSON
    with st.expander("View Raw JSON State"):
        st.code(st.session_state.non_fiction_state.model_dump_json(indent=2), language="json")

if __name__ == "__main__":
    main()
