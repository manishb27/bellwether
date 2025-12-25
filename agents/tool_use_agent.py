import os
import json
import difflib
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field
from openai import OpenAI
import inspect

# ============================================================
# 1. PYDANTIC MODELS (The Source of Truth)
# ============================================================

class Character(BaseModel):
    name: str = Field(..., description="Primary name of the character")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/titles")
    appearance: Optional[str] = Field(None, description="Concise visual description (under 30 words)")
    traits: List[str] = Field(default_factory=list, description="Stable personality traits")
    relationships: dict[str, str] = Field(default_factory=dict, description="Map of {other_char: relationship_type}")
    evidence: List[str] = Field(default_factory=list, description="Direct quotes supporting key details")

class Location(BaseModel):
    name: str = Field(..., description="Name of the location")
    description: Optional[str] = Field(None, description="Visual description of the setting")
    evidence: List[str] = Field(default_factory=list)

class PlotPoint(BaseModel):
    description: str = Field(..., description="A distinct, major event in the narrative")
    evidence: List[str] = Field(default_factory=list)



class FictionState(BaseModel):
    characters: List[Character] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    plot_points: List[PlotPoint] = Field(default_factory=list)

    def get_summary(self):
        """Returns a token-optimized summary for the LLM context."""
        summary = {
            "characters": [
                {
                    "name": c.name,
                    "aliases": c.aliases,
                    "appearance": c.appearance,
                    "traits": c.traits,
                    "relationships": c.relationships
                } for c in self.characters
            ],
            "locations": [
                {
                    "name": l.name,
                    "description": l.description
                } for l in self.locations
            ],
            # Only show description for plot points to save tokens
            "plot_points": [p.description for p in self.plot_points]
        }
        return json.dumps(summary)


# ============================================================
# 2. HELPER FUNCTIONS (Logic & Hardening)
# ============================================================

def is_similar_string(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Uses SequenceMatcher to check if two strings are roughly the same."""
    return difflib.SequenceMatcher(None, s1, s2).ratio() > threshold

def normalize_string(s: str) -> str:
    return s.strip().lower()

def safe_append_unique(target_list: List[str], new_item: str):
    """Appends to list only if a normalized version doesn't exist."""
    norm_new = normalize_string(new_item)
    for item in target_list:
        if normalize_string(item) == norm_new:
            return # Duplicate found
    target_list.append(new_item)


# ============================================================
# 3. THE TOOL DEFINITION
# ============================================================

class NewCharacterArg(BaseModel):
    name: str = Field(..., description="Name of the character")
    aliases: List[str] = Field(default_factory=list, description="Other names or titles")
    appearance: Optional[str] = Field(None, description="Visual appearance. concise.")
    traits: List[str] = Field(default_factory=list, description="Personality traits (adjectives)")
    relationships: dict[str, str] = Field(default_factory=dict, description="Relationships to others")
    evidence: List[str] = Field(default_factory=list, description="One key quote")

class UpdateCharacterArg(BaseModel):
    name: str = Field(..., description="Name of the character to update (fuzzy match allowed)")
    aliases: List[str] = Field(default_factory=list, description="New aliases")
    appearance: Optional[str] = Field(None, description="Updated visual description")
    traits: List[str] = Field(default_factory=list, description="New traits")
    relationships: dict[str, str] = Field(default_factory=dict, description="New relationships")
    evidence: List[str] = Field(default_factory=list, description="New evidence")

class NewLocationArg(BaseModel):
    name: str = Field(..., description="Name of the location")
    description: str = Field(..., description="Visual description")
    evidence: List[str] = Field(default_factory=list)

class UpdateLocationArg(BaseModel):
    name: str = Field(..., description="Name of the location to update")
    description: Optional[str] = Field(None, description="New description details")
    evidence: List[str] = Field(default_factory=list)

class NewPlotPointArg(BaseModel):
    description: str = Field(..., description="Detailed description of the event")
    evidence: List[str] = Field(default_factory=list)

class UpdateKnowledgeBaseArgs(BaseModel):
    new_characters: List[NewCharacterArg] = Field(default_factory=list, description="Add NEW characters only.")
    updated_characters: List[UpdateCharacterArg] = Field(default_factory=list, description="Update EXISTING characters.")
    new_locations: List[NewLocationArg] = Field(default_factory=list, description="Add NEW locations.")
    updated_locations: List[UpdateLocationArg] = Field(default_factory=list, description="Update EXISTING locations.")
    new_plot_points: List[NewPlotPointArg] = Field(default_factory=list, description="Add SIGNIFICANT new plot events.")

def update_knowledge_base(state: FictionState, args: UpdateKnowledgeBaseArgs) -> str:
    """
    Refined logic for merging state updates with deduplication.
    """
    print(f"DEBUG ARGS: {args}")
    changes_log = []

    # --- 1. CHARACTERS ---
    existing_char_map = {c.name.lower(): c for c in state.characters}
    # Also map aliases for robust lookup
    alias_map = {}
    for c in state.characters:
        for a in c.aliases:
            alias_map[a.lower()] = c

    # A. New Characters
    for char_arg in args.new_characters:
        name_lower = char_arg.name.lower()
        if name_lower in existing_char_map or name_lower in alias_map:
            # It already exists, skip creation
            continue
        
        new_char = Character(
            name=char_arg.name,
            aliases=char_arg.aliases,
            appearance=char_arg.appearance,
            traits=char_arg.traits,
            relationships=char_arg.relationships,
            evidence=char_arg.evidence
        )
        state.characters.append(new_char)
        existing_char_map[name_lower] = new_char # Update local map
        changes_log.append(f"Created char '{char_arg.name}'")

    # B. Update Characters
    for update_arg in args.updated_characters:
        name_lower = update_arg.name.lower()
        target = existing_char_map.get(name_lower) or alias_map.get(name_lower)
        
        if target:
            # Update Appearance (Overwrite if new is longer/better, otherwise ignore)
            if update_arg.appearance:
                if not target.appearance or len(update_arg.appearance) > len(target.appearance):
                    target.appearance = update_arg.appearance

            # Merge Aliases
            for a in update_arg.aliases:
                safe_append_unique(target.aliases, a)
            
            # Merge Traits
            for t in update_arg.traits:
                safe_append_unique(target.traits, t)
            
            # Merge Relationships
            if update_arg.relationships:
                target.relationships.update(update_arg.relationships)

            # Merge Evidence
            for e in update_arg.evidence:
                safe_append_unique(target.evidence, e)
            
            changes_log.append(f"Updated char '{target.name}'")

    # --- 2. LOCATIONS ---
    existing_loc_map = {l.name.lower(): l for l in state.locations}

    # A. New Locations
    for loc_arg in args.new_locations:
        if loc_arg.name.lower() not in existing_loc_map:
            new_loc = Location(
                name=loc_arg.name,
                description=loc_arg.description,
                evidence=loc_arg.evidence
            )
            state.locations.append(new_loc)
            existing_loc_map[loc_arg.name.lower()] = new_loc
            changes_log.append(f"Created location '{loc_arg.name}'")

    # B. Update Locations
    for loc_upd in args.updated_locations:
        target_loc = existing_loc_map.get(loc_upd.name.lower())
        if target_loc:
            if loc_upd.description:
                 if not target_loc.description or len(loc_upd.description) > len(target_loc.description):
                    target_loc.description = loc_upd.description
            for e in loc_upd.evidence:
                safe_append_unique(target_loc.evidence, e)
            changes_log.append(f"Updated location '{target_loc.name}'")

    # --- 3. PLOT POINTS (Deduplication Logic) ---
    for pp_arg in args.new_plot_points:
        # Check against existing plot points using fuzzy matching
        is_duplicate = False
        for existing_pp in state.plot_points:
            if is_similar_string(existing_pp.description, pp_arg.description, threshold=0.75):
                # It's duplicate. Just append evidence if new.
                for e in pp_arg.evidence:
                    safe_append_unique(existing_pp.evidence, e)
                is_duplicate = True
                break
        
        if not is_duplicate:
            state.plot_points.append(PlotPoint(
                description=pp_arg.description,
                evidence=pp_arg.evidence
            ))
            changes_log.append("Added plot point")
    
    if not changes_log:
        return "No significant changes made (deduplication active)."
    
    return f"Success. Changes: {', '.join(changes_log)}"

# ============================================================
# 4. THE AGENT (With Improved Prompting)
# ============================================================

class Agent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
    
    def process_chunk(self, state: FictionState, text_chunk: str):
        # 1. Check if tool is available? Yes.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_knowledge_base",
                    "description": "Updates the fiction knowledge base. Call this ONLY if you find NEW info.",
                    "parameters": UpdateKnowledgeBaseArgs.model_json_schema()
                }
            }
        ]
        
        # 2. System Prompt (The Editor Persona)
        system_prompt = (
            "You are a Senior Fiction Editor building a Series Bible. "
            "Your goal is to extract FACTUAL, STRUCTURAL information from the text.\n\n"
            "GUIDELINES:\n"
            "1. CHARACTERS: identify names, tangible traits, and relationships. Ignore temporary emotions.\n"
            "2. PLOT: Record distinct narrative beats (actions that change the story state). Ignore atmospheric descriptions.\n"
            "3. LOCATIONS: Track setting changes.\n"
            "4. DEDUPLICATION: Do not re-add information that already exists in the 'Current State Summary'.\n"
            "5. EVIDENCE: Provide 1 brief quote per fact. Do not copy long paragraphs."
        )

        # 3. Send Message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current State Summary:\n{state.get_summary()}\n\nNew Text Chunk:\n{text_chunk}"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" 
        )

        message = response.choices[0].message
        
        # 4. Handle Tool Call
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "update_knowledge_base":
                    print(" >> Agent decided to call tool: update_knowledge_base")
                    # Parse arguments
                    args_data = json.loads(tool_call.function.arguments)
                    try:
                        tool_args = UpdateKnowledgeBaseArgs(**args_data)
                        # Execute Tool
                        result_msg = update_knowledge_base(state, tool_args)
                        print(f" >> Tool Execution: {result_msg}")
                    except Exception as e:
                        print(f"Tool Error: {e}")
        else:
            print(" >> Agent decided NOT to call any tool (No new info found).")

    def resolve_entities(self, state: FictionState) -> str:
        """
        Analyzes the current character list and merges duplicates (e.g. 'The King' -> 'King Harold').
        """
        if len(state.characters) < 2:
            return "Not enough characters to merge."

        # 1. Define strict schema for the decision
        class MergePair(BaseModel):
            keep_name: str = Field(..., description="The more complete/official name (e.g. 'King Harold')")
            merge_name: str = Field(..., description="The duplicate or alias name to remove (e.g. 'The King')")
            reason: str

        class ResolutionResult(BaseModel):
            merges: List[MergePair] = Field(default_factory=list)

        # 2. Prompt
        char_list = [f"- {c.name} (Aliases: {c.aliases})" for c in state.characters]
        prompt = (
             "Analyze this list of characters. Identify any entries that refer to the SAME person.\n"
             "Return a list of pairs to merge. If no duplicates exist, return an empty list.\n\n"
             f"Characters:\n{chr(10).join(char_list)}"
        )

        try:
            # 3. Call LLM with Structured Output
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=ResolutionResult
            )
            result = response.choices[0].message.parsed
            
            if not result.merges:
                return "No duplicates found."
            
            # 4. Execute Merges
            log = []
            for pair in result.merges:
                success = self._execute_merge(state, pair.keep_name, pair.merge_name)
                if success:
                    log.append(f"Merged '{pair.merge_name}' into '{pair.keep_name}'")
            
            return ", ".join(log) if log else "No merges executed."

        except Exception as e:
            return f"Resolution failed: {e}"

    def _execute_merge(self, state: FictionState, keep_name: str, merge_name: str) -> bool:
        """Helper to physically merge two character records."""
        # Find objects
        keep_char = next((c for c in state.characters if c.name.lower() == keep_name.lower()), None)
        merge_char = next((c for c in state.characters if c.name.lower() == merge_name.lower()), None)

        if not keep_char or not merge_char:
            return False
            
        # 1. Aliases: Add the merged name itself + its aliases
        if merge_char.name not in keep_char.aliases:
            keep_char.aliases.append(merge_char.name)
        for a in merge_char.aliases:
            if a not in keep_char.aliases:
                keep_char.aliases.append(a)
        
        # 2. Appearance (Keep longest)
        if merge_char.appearance:
            if not keep_char.appearance or len(merge_char.appearance) > len(keep_char.appearance):
                keep_char.appearance = merge_char.appearance
        
        # 3. Traits
        for t in merge_char.traits:
            if t not in keep_char.traits:
                keep_char.traits.append(t)
                
        # 4. Evidence
        for e in merge_char.evidence:
             if e not in keep_char.evidence:
                 keep_char.evidence.append(e)

        # 5. Relationships: Update and Merge
        keep_char.relationships.update(merge_char.relationships) 
        
        # 6. Remove the old char
        state.characters.remove(merge_char)
        
        # 7. Update OTHER characters' relationships that pointed to the old name
        for c in state.characters:
            if merge_name in c.relationships:
                rel_type = c.relationships.pop(merge_name)
                c.relationships[keep_name] = rel_type
                
        return True




# ============================================================
# 5. MAIN FLOW
# ============================================================

if __name__ == "__main__":
    print("--- Tool-Use Agent Demo ---")
    
    agent = Agent()
    state = FictionState()

    chunk1 = """
    The door creaked open and Mira stepped into the abandoned observatory, her silver hair catching the faint starlight.
    She hesitated — the message from Elden had been urgent, almost desperate.
    “Mira,” Elden whispered from the shadows, tall and wrapped in a weather-worn black coat.
    """
    
    print(f"\nProcessing Chunk 1...")
    agent.process_chunk(state, chunk1)
    print("State Snapshot:", state.model_dump_json(indent=2))

    chunk2 = """
    Mira moved quickly across the frost-covered bridge.
    She had always feared the royal court, but news of the king's poisoning had shaken her.
    Ahead of them, a young boy named Corin emerged, clutching a sealed parchment.
    """

    print(f"\nProcessing Chunk 2...")
    agent.process_chunk(state, chunk2)
    print("State Snapshot:", state.model_dump_json(indent=2))
