import os
import json
import difflib
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from openai import OpenAI

# ============================================================
# 1. PYDANTIC MODELS (The Source of Truth)
# ============================================================

class Concept(BaseModel):
    name: str = Field(..., description="Name of the concept")
    centrality: Literal["core", "supporting", "contextual"] = Field("contextual", description="Importance of the concept")
    definition: str = Field(..., description="Author's definition/framing")
    scope: Optional[str] = Field(None, description="Deprecated. Use facets.")
    # Scope is now handled via new facets rather than just a string to prevent drift
    facets: List[str] = Field(default_factory=list, description="Additional aspects, nuances, or scope details appended over time")
    related_concepts: List[str] = Field(default_factory=list, description="List of related concept names")
    evidence_quotes: List[str] = Field(default_factory=list, description="Verbatim text spans")

class Claim(BaseModel):
    claim_text: str = Field(..., description="The canonical core assertion")
    variations: List[str] = Field(default_factory=list, description="Paraphrases or slightly different phrasings of the same claim")
    claim_type: Literal["descriptive", "causal", "comparative", "prescriptive", "predictive"] = Field(..., description="Type of claim")
    status: Literal["explicit", "implied"] = Field(..., description="Explicitly stated or implied")
    confidence: Literal["strong", "moderate", "weak"] = Field(..., description="Author's confidence based on hedging and evidence")
    dependencies: List[str] = Field(default_factory=list, description="Other claims this depends on or supports")
    supporting_evidence_refs: List[str] = Field(default_factory=list, description="References to evidence")

class Evidence(BaseModel):
    content: str = Field(..., description="The core content of the evidence")
    evidence_type: Literal["data_statistics", "experiment", "case_study", "historical_example", "expert_testimony", "analogy"] = Field(..., description="Type of evidence")
    supports_claim: Optional[str] = Field(None, description="The canonical claim text this evidence supports")
    source_quote: str = Field(..., description="Verbatim quote")
    strength: Literal["strong", "moderate", "weak"] = Field(..., description="Strength of evidence")

class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    category: str = Field(..., description="e.g. Organization, Person, Technology")
    role: str = Field(..., description="Role in the text")
    relationships: Dict[str, str] = Field(default_factory=dict, description="Map of {other_entity: relationship}")

class Process(BaseModel):
    name: str = Field(..., description="Name of the executable mechanism or flow")
    steps: List[str] = Field(default_factory=list, description="Ordered executable steps")
    inputs: List[str] = Field(default_factory=list, description="Inputs")
    outputs: List[str] = Field(default_factory=list, description="Outputs")
    conditions: List[str] = Field(default_factory=list, description="Conditions or failure modes")

class Assumption(BaseModel):
    assumption_text: str = Field(..., description="Condition that must be true (stated or unstated)")
    scope: Optional[str] = Field(None, description="Scope where this holds")
    acknowledged: bool = Field(..., description="Whether explicitly acknowledged by author")
    risk_if_invalid: Optional[str] = Field(None, description="Risk if assumption is wrong")

class Implication(BaseModel):
    implication_text: str = Field(..., description="Outcome if claims are accepted")
    dependent_claims: List[str] = Field(default_factory=list, description="Claims that lead to this implication")
    affected_entities: List[str] = Field(default_factory=list, description="Entities affected")

class NonFictionState(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    processes: List[Process] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)
    implications: List[Implication] = Field(default_factory=list)

    def get_summary(self):
        """Returns a token-optimized summary for the LLM context."""
        summary = {
            "concepts": [f"{c.name} ({c.centrality})" for c in self.concepts],
            "claims_count": len(self.claims),
            "evidence_count": len(self.evidence),
            "entities": [e.name for e in self.entities],
            "processes": [p.name for p in self.processes],
            "assumptions_count": len(self.assumptions),
            "implications_count": len(self.implications)
        }
        return json.dumps(summary)
    
    def get_pretty_view(self) -> str:
        """Returns a human-readable string representation of the knowledge base."""
        lines = ["=== NON-FICTION KNOWLEDGE BASE ===", ""]
        
        if self.concepts:
            lines.append("## CONCEPTS")
            sorted_concepts = sorted(self.concepts, key=lambda x: 0 if x.centrality == 'core' else 1)
            for c in sorted_concepts:
                lines.append(f"- **{c.name}** [{c.centrality.upper()}]: {c.definition}")
                for facet in c.facets:
                     lines.append(f"  * {facet}")
            lines.append("")

        if self.claims:
            lines.append("## CLAIMS")
            for c in self.claims:
                lines.append(f"- [{c.claim_type.upper()}] {c.claim_text} ({c.confidence})")
                if c.dependencies:
                    lines.append(f"  * Depends on/Supports: {', '.join(c.dependencies)}")
                if c.variations:
                    lines.append(f"  * Variations: {len(c.variations)} stored")
            lines.append("")

        if self.evidence:
            lines.append("## EVIDENCE")
            for e in self.evidence:
                lines.append(f"- [{e.evidence_type.replace('_', ' ').title()}] {e.content} (Strength: {e.strength})")
                if e.supports_claim:
                    lines.append(f"  Supports: '{e.supports_claim}'")
            lines.append("")

        if self.entities:
            lines.append("## ENTITIES")
            for e in self.entities:
                lines.append(f"- **{e.name}** ({e.category}): {e.role}")
            lines.append("")

        if self.processes:
            lines.append("## PROCESSES")
            for p in self.processes:
                lines.append(f"- **{p.name}**")
                for i, step in enumerate(p.steps, 1):
                    lines.append(f"  {i}. {step}")
            lines.append("")

        if self.assumptions:
            lines.append("## ASSUMPTIONS")
            for a in self.assumptions:
                ack = "Acknowledged" if a.acknowledged else "Unacknowledged"
                lines.append(f"- {a.assumption_text} ({ack})")
            lines.append("")

        if self.implications:
            lines.append("## IMPLICATIONS")
            for i in self.implications:
                lines.append(f"- {i.implication_text}")
                if i.affected_entities:
                    lines.append(f"  Affects: {', '.join(i.affected_entities)}")
            lines.append("")
            
        return "\n".join(lines)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_string(s: str) -> str:
    return s.strip().lower()

def safe_append_unique(target_list: List[str], new_item: str):
    """Appends to list only if a normalized version doesn't exist."""
    norm_new = normalize_string(new_item)
    for item in target_list:
        if normalize_string(item) == norm_new:
            return 
    target_list.append(new_item)

def is_similar_string(s1: str, s2: str, threshold: float = 0.85) -> bool:
    """Uses SequenceMatcher to check if two strings are roughly the same."""
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio() > threshold


# ============================================================
# 3. TOOL ARGUMENTS DEFINITION
# ============================================================

class NewConceptArg(BaseModel):
    name: str
    centrality: Literal["core", "supporting", "contextual"]
    definition: str
    facets: List[str] = []
    related_concepts: List[str] = []
    evidence_quote: str

class UpdateConceptArg(BaseModel):
    name: str = Field(..., description="Existing concept name to update")
    add_facets: List[str] = []
    add_related_concepts: List[str] = []
    add_evidence_quote: Optional[str] = None
    # NOTE: No definition update allowed to prevent drift, only facets

class NewClaimArg(BaseModel):
    claim_text: str
    claim_type: Literal["descriptive", "causal", "comparative", "prescriptive", "predictive"]
    status: Literal["explicit", "implied"]
    confidence: Literal["strong", "moderate", "weak"]
    dependencies: List[str] = []
    evidence_quote: str

class UpdateClaimArg(BaseModel):
    claim_text_snippet: str
    add_variation: Optional[str] = None # For paraphrases
    add_evidence_quote: Optional[str] = None

class NewEvidenceArg(BaseModel):
    content: str
    evidence_type: Literal["data_statistics", "experiment", "case_study", "historical_example", "expert_testimony", "analogy"]
    supports_claim: Optional[str] = Field(None, description="Text of the claim this supports")
    source_quote: str
    strength: Literal["strong", "moderate", "weak"]

class NewEntityArg(BaseModel):
    name: str
    category: str
    role: str
    relationships: Dict[str, str] = {}

class NewProcessArg(BaseModel):
    name: str
    steps: List[str]
    inputs: List[str] = []
    outputs: List[str] = []
    conditions: List[str] = []

class NewAssumptionArg(BaseModel):
    assumption_text: str
    scope: Optional[str] = None
    acknowledged: bool
    risk: Optional[str] = None

class NewImplicationArg(BaseModel):
    implication_text: str
    dependent_claims: List[str] = []
    affected_entities: List[str] = []

class UpdateKnowledgeBaseArgs(BaseModel):
    new_concepts: List[NewConceptArg] = []
    updated_concepts: List[UpdateConceptArg] = []
    
    new_claims: List[NewClaimArg] = []
    updated_claims: List[UpdateClaimArg] = []
    
    new_evidence: List[NewEvidenceArg] = []
    
    new_entities: List[NewEntityArg] = []
    
    new_processes: List[NewProcessArg] = []
    
    new_assumptions: List[NewAssumptionArg] = []
    
    new_implications: List[NewImplicationArg] = []


def update_knowledge_base(state: NonFictionState, args: UpdateKnowledgeBaseArgs) -> str:
    """
    Executes the tool logic to update the state object.
    """
    changes_log = []

    # --- CONCEPTS ---
    concept_map = {normalize_string(c.name): c for c in state.concepts}
    
    for nc in args.new_concepts:
        if normalize_string(nc.name) in concept_map:
            continue 
        new_c = Concept(
            name=nc.name,
            centrality=nc.centrality,
            definition=nc.definition,
            facets=nc.facets,
            related_concepts=nc.related_concepts,
            evidence_quotes=[nc.evidence_quote]
        )
        state.concepts.append(new_c)
        concept_map[normalize_string(nc.name)] = new_c
        changes_log.append(f"Added Concept: {nc.name}")

    for uc in args.updated_concepts:
        target = concept_map.get(normalize_string(uc.name))
        if target:
            # We do NOT overwrite definition here to prevent drift
            for f in uc.add_facets:
                safe_append_unique(target.facets, f)
            for rc in uc.add_related_concepts:
                safe_append_unique(target.related_concepts, rc)
            if uc.add_evidence_quote:
                safe_append_unique(target.evidence_quotes, uc.add_evidence_quote)
            changes_log.append(f"Updated Concept: {target.name}")

    # --- CLAIMS ---
    for ncl in args.new_claims:
        unique = True
        for ex in state.claims:
            if is_similar_string(ex.claim_text, ncl.claim_text):
                unique = False
                # Canonicalization: If nearly identical, just add as variation/evidence
                safe_append_unique(ex.variations, ncl.claim_text)
                changes_log.append(f"Merged claim variant into '{ex.claim_text[:20]}...'")
                break
        if unique:
            state.claims.append(Claim(
                claim_text=ncl.claim_text,
                claim_type=ncl.claim_type,
                status=ncl.status,
                confidence=ncl.confidence,
                dependencies=ncl.dependencies,
                supporting_evidence_refs=[ncl.evidence_quote]
            ))
            changes_log.append("Added Claim")

    for ucl in args.updated_claims:
        for ex in state.claims:
            if ucl.claim_text_snippet.lower() in ex.claim_text.lower():
                if ucl.add_variation:
                     safe_append_unique(ex.variations, ucl.add_variation)
                changes_log.append("Updated Claim")
                break

    # --- EVIDENCE ---
    for ne in args.new_evidence:
        state.evidence.append(Evidence(
            content=ne.content,
            evidence_type=ne.evidence_type,
            supports_claim=ne.supports_claim,
            source_quote=ne.source_quote,
            strength=ne.strength
        ))
        changes_log.append("Added Evidence")

    # --- ENTITIES ---
    ent_map = {normalize_string(e.name): e for e in state.entities}
    for ne in args.new_entities:
        if normalize_string(ne.name) in ent_map:
            continue
        new_e = Entity(
            name=ne.name,
            category=ne.category,
            role=ne.role,
            relationships=ne.relationships
        )
        state.entities.append(new_e)
        ent_map[normalize_string(ne.name)] = new_e
        changes_log.append(f"Added Entity: {ne.name}")

    # --- PROCESSES ---
    for np in args.new_processes:
        unique = True
        for p in state.processes:
            if normalize_string(p.name) == normalize_string(np.name):
                unique = False
                break
        if unique:
            state.processes.append(Process(
                name=np.name,
                steps=np.steps,
                inputs=np.inputs,
                outputs=np.outputs,
                conditions=np.conditions
            ))
            changes_log.append(f"Added Process: {np.name}")

    # --- ASSUMPTIONS ---
    for na in args.new_assumptions:
        state.assumptions.append(Assumption(
            assumption_text=na.assumption_text,
            scope=na.scope,
            acknowledged=na.acknowledged,
            risk_if_invalid=na.risk
        ))
        changes_log.append("Added Assumption")

    # --- IMPLICATIONS ---
    for ni in args.new_implications:
        state.implications.append(Implication(
            implication_text=ni.implication_text,
            dependent_claims=ni.dependent_claims,
            affected_entities=ni.affected_entities
        ))
        changes_log.append("Added Implication")

    if not changes_log:
        return "No changes made (deduplication active)."
    
    return f"Success. Log: {', '.join(changes_log)}"


# ============================================================
# 4. THE AGENT CLASS
# ============================================================

class Agent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def process_chunk(self, state: NonFictionState, text_chunk: str):
        # Tools configuration
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_knowledge_base",
                    "description": "Updates the knowledge base with extracted non-fiction primitives.",
                    "parameters": UpdateKnowledgeBaseArgs.model_json_schema()
                }
            }
        ]

        # System Prompt
        system_prompt = (
            "You are a Senior Knowledge Architect and Non-Fiction Editor.\n"
            "Your task is to REFINE a generalized extraction system. Fix structural errors.\n\n"
            "=== STRUCTURAL REQUIREMENTS ===\n"
            "1. CONCEPT CENTRALITY: Classify as Core | Supporting | Contextual. Contextual concepts must NOT be primary.\n"
            "2. CONCEPT DRIFT: Preserve core definitions. Append new facets/nuances instead of overwriting.\n"
            "3. CLAIM CANONICALIZATION: Merge equivalent claims. Store paraphrases as variations. Avoid duplicate assertions.\n"
            "4. PROCESS vs PRACTICE: Processes must be executable mechanisms. Normative advice is NOT a process.\n"
            "5. ASSUMPTIONS: Extract implicit assumptions required for claims to hold. Treat unstated premises as first-class.\n"
            "6. CONFIDENCE: Infer based on evidence/hedging. Do not default to 'strong'.\n"
            "7. ARGUMENTS: Capture 'depends_on' relationships.\n\n"
            "=== EXTRACTION CATEGORIES ===\n"
            "- CONCEPTS\n"
            "- CLAIMS (types: descriptive, causal, comparative, prescriptive, predictive)\n"
            "- EVIDENCE (types: data, experiment, case_study, historical, expert, analogy)\n"
            "- ENTITIES\n"
            "- PROCESSES (executable steps only)\n"
            "- ASSUMPTIONS (stated or unstated)\n"
            "- IMPLICATIONS\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current State Summary:\n{state.get_summary()}\n\nNew Text Chunk:\n{text_chunk}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "update_knowledge_base":
                        print(" >> Extracting Knowledge...")
                        args_data = json.loads(tool_call.function.arguments)
                        tool_args = UpdateKnowledgeBaseArgs(**args_data)
                        result = update_knowledge_base(state, tool_args)
                        print(f" >> {result}")
            else:
                print(" >> No new knowledge found in this chunk.")
                
        except Exception as e:
            print(f"Error processing chunk: {e}")

    def resolve_entities(self, state: NonFictionState) -> str:
        """
        Secondary pass to merge duplicate Concepts or Entities.
        """
        items = []
        for c in state.concepts:
            items.append(f"CONCEPT: {c.name} (Def: {c.definition[:50]}...)")
        for e in state.entities:
            items.append(f"ENTITY: {e.name} (Role: {e.role[:50]}...)")

        if len(items) < 2:
            return "Not enough items to resolve."

        class MergePair(BaseModel):
            keep_name: str
            type: Literal["CONCEPT", "ENTITY"]
            merge_names: List[str]
            reason: str

        class ResolutionResult(BaseModel):
            merges: List[MergePair]

        prompt = (
            "Analyze this list of knowledge base items. Identify if any distinct entries refer to the SAME person/idea "
            "and should be merged (duplicates or synonyms).\n\n"
            "Items:\n" + "\n".join(items)
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=ResolutionResult
            )
            result = response.choices[0].message.parsed

            if not result.merges:
                return "No duplicates found."

            log = []
            for m in result.merges:
                # Execute merge logic
                if m.type == "CONCEPT":
                    self._merge_concepts(state, m.keep_name, m.merge_names)
                elif m.type == "ENTITY":
                    self._merge_entities(state, m.keep_name, m.merge_names)
                log.append(f"Merged {m.merge_names} into {m.keep_name}")

            return ", ".join(log)

        except Exception as e:
            return f"Resolution error: {e}"

    def _merge_concepts(self, state: NonFictionState, keep_name: str, merge_names: List[str]):
        keep_obj = next((x for x in state.concepts if normalize_string(x.name) == normalize_string(keep_name)), None)
        if not keep_obj: return

        for mn in merge_names:
            merge_obj = next((x for x in state.concepts if normalize_string(x.name) == normalize_string(mn)), None)
            if not merge_obj or merge_obj == keep_obj: continue
            
            # Definition is NOT merged to preserve drift protection, but facets are
            keep_obj.facets.extend(merge_obj.facets) 
            keep_obj.related_concepts.extend(merge_obj.related_concepts)
            keep_obj.evidence_quotes.extend(merge_obj.evidence_quotes)
            state.concepts.remove(merge_obj)

    def _merge_entities(self, state: NonFictionState, keep_name: str, merge_names: List[str]):
        keep_obj = next((x for x in state.entities if normalize_string(x.name) == normalize_string(keep_name)), None)
        if not keep_obj: return

        for mn in merge_names:
            merge_obj = next((x for x in state.entities if normalize_string(x.name) == normalize_string(mn)), None)
            if not merge_obj or merge_obj == keep_obj: continue
            
            keep_obj.relationships.update(merge_obj.relationships)
            state.entities.remove(merge_obj)

if __name__ == "__main__":
    print("Agent updated.")
