import os
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# ============================================================
# 1. PYDANTIC MODELS (Grammar Specific)
# ============================================================

class GrammarIssue(BaseModel):
    type: str = Field(..., description="Category: 'grammar', 'clarity', or 'style'")
    issue: str = Field(..., description="The name of the error (e.g. 'Dangling Modifier')")
    explanation: str = Field(..., description="Brief 'Why this matters' explanation")
    original_text: str = Field(..., description="Exact excerpt containing the issue")
    suggested_fix: str = Field(..., description="Improved version")
    severity: str = Field(..., description="'high' (critical error), 'medium' (confusing), 'low' (suggestion)")
    confidence: str = Field("medium", description="'high' or 'medium'")

class GrammarReport(BaseModel):
    grammar_issues: List[GrammarIssue] = Field(default_factory=list)

# ============================================================
# 2. LOGIC
# ============================================================

class GrammarAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def analyze_chunk(self, text_chunk: str) -> GrammarReport:
        """
        Analyzes a text chunk for grammar, clarity, and style issues using an LLM.
        Does NOT modify the text, only reports issues.
        """
        system_prompt = (
            "You are a Senior Fiction Copyeditor. Your job is to improve clarity without killing the author's voice.\n"
            "GUIDELINES:\n"
            "1. BE CONSERVATIVE: Ignore stylistic fragments, dialect, or poetic license unless it causes genuine confusion.\n"
            "2. FOCUS ON: Objective errors (typos, subject-verb), dangling modifiers, unclear antecedents, and unintentional repetition.\n"
            "3. SEVERITY:\n"
            "   - 'high': Real errors (grammar/spelling) or unintelligible sentences.\n"
            "   - 'medium': Awkward phrasing or ambiguity.\n"
            "   - 'low': Minor suggestions (word choice, tightness).\n"
            "4. EXPLAIN: Provide a brief 1-sentence explanation for *why* this is an issue."
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{text_chunk}"}
                ],
                response_format=GrammarReport
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Grammar Analysis Error: {e}")
            return GrammarReport()
