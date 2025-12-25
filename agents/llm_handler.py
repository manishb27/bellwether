import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

class LLMIssue(BaseModel):
    issue: str = Field(..., description="Name of the issue")
    explanation: str = Field(..., description="Brief explanation")
    original_text: str = Field(..., description="Exact text match")
    suggested_fix: str = Field(..., description="Suggestion")
    severity: str = Field("medium", description="high, medium, or low")
    type: str = Field("style", description="grammar, clarity, style")

class LLMReport(BaseModel):
    issues: List[LLMIssue] = Field(default_factory=list)

class LLMHandler:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def analyze_contextual(self, text: str) -> List[LLMIssue]:
        """
        Uses LLM to catch subtle contextual errors (Affect/Effect) 
        and high-level improvements that rule-based tools miss.
        """
        system_prompt = (
            "You are a Senior Fiction Copyeditor. Your job is to catch subtle contextual errors, stylistic weaknesses, and grammar issues that rule-based tools might miss.\n"
            "TARGETS:\n"
            "1. **Contextual Spelling & Homophones** (affect/effect, their/there, peak/peek).\n"
            "2. **Grammar & Agreement**:\n"
            "   - Subject-Verb Agreement mismatches (e.g., 'The team of players are running' -> 'is running').\n"
            "   - Determiner-Noun mismatches (e.g., 'Those kind of things' -> 'That kind').\n"
            "3. **Stylistic Weaknesses**:\n"
            "   - **Passive Voice**: Flag unnecessary passive voice usage (e.g., 'The ball was thrown by him').\n"
            "   - **Repetitive Structure**.\n"
            "4. **Semantic Conflicts** (e.g. 'He nodded his head no').\n"
            "5. **Exclusions**: DO NOT report simple spelling errors (typos) as they are handled by a separate system. Focus on context.\n"
            "6. ONLY report true issues. If text is fine, return empty list."
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text for subtle contextual errors:\n\n{text}"}
                ],
                response_format=LLMReport,
                temperature=0
            )
            return response.choices[0].message.parsed.issues
        except Exception as e:
            print(f"LLM Handler Error: {e}")
            return []
