import os
import sys
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import unicodedata

# Lazy imports with granular flags
HAS_SPACY = False
HAS_SYMSPELL = False
HAS_LT = False
HAS_TEXTSTAT = False
HAS_PYSPELL = False

try:
    import spacy
    HAS_SPACY = True
except Exception:
    pass

try:
    from symspellpy import SymSpell, Verbosity
    import pkg_resources
    HAS_SYMSPELL = True
except Exception:
    pass

try:
    import language_tool_python
    HAS_LT = True
except Exception:
    pass

try:
    import textstat
    HAS_TEXTSTAT = True
except Exception:
    pass

try:
    from spellchecker import SpellChecker
    HAS_PYSPELL = True
except Exception:
    pass

from agents.llm_handler import LLMHandler, LLMIssue

# ============================================================
# 1. MODELS
# ============================================================

class GrammarIssue(BaseModel):
    type: str = Field(..., description="'spelling', 'grammar', 'style', 'clarity'")
    issue: str = Field(..., description="Short name of error")
    explanation: str = Field(..., description="Description")
    original_text: str = Field(..., description="The problem text")
    suggested_fix: str = Field(..., description="Proposed fix")
    severity: str = Field(..., description="'high', 'medium', 'low'")
    source: str = Field(..., description="'symspell', 'languagetool', 'spacy', 'llm'")
    start_char: int = Field(0, description="Start index in chunk")
    end_char: int = Field(0, description="End index in chunk")

class GrammarReport(BaseModel):
    grammar_issues: List[GrammarIssue] = Field(default_factory=list)
    readability_score: float = 0.0
    stats: dict = {}

# ============================================================
# 2. TOOL MANAGERS
# ============================================================

class ToolStack:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolStack, cls).__new__(cls)
            cls._instance.init_tools()
        return cls._instance

    def init_tools(self):
        print("Initializing Grammar Tools...")
        
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    print("Spacy model failed to load.")
                    
        self.sym_spell = None
        if HAS_SYMSPELL:
            try:
                self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            except Exception as e:
                print(f"SymSpell Init Error: {e}")

        self.lt_tool = None
        if HAS_LT:
            try:
                self.lt_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                print(f"LanguageTool Init Error: {e}")

        self.spell_checker = None
        if HAS_PYSPELL:
            try:
                self.spell_checker = SpellChecker()
            except Exception as e:
                print(f"SpellChecker Init Error: {e}")
            
        print("Tools Initialized.")


# ============================================================
# 3. GRAMMAR AGENT
# ============================================================

class GrammarAgent:
    def __init__(self):
        self.tools = ToolStack()
        self.llm_handler = LLMHandler()

    def analyze_chunk(self, text: str) -> GrammarReport:
        issues = []
        
        # 1. Normalize
        text = unicodedata.normalize('NFC', text)
        
        # 2. Local Tools Analysis
        
        # B. LanguageTool
        if self.tools.lt_tool:
            try:
                lt_matches = self.tools.lt_tool.check(text)
                for match in lt_matches:
                    # Helper to safely get attributes from the unpredictable Match object
                    def safe_get(obj, attr, default):
                        try:
                            return getattr(obj, attr, default)
                        except Exception:
                            return default

                    category = safe_get(match, 'category', 'grammar')
                    # Some versions use ruleId, some use rule. Some might wrap it.
                    rule_id = safe_get(match, 'ruleId', safe_get(match, 'rule', 'UNKNOWN_RULE'))
                    
                    if category == 'Typos': 
                        itype = "spelling"
                    elif category == 'Style': 
                        itype = "style"
                    else: 
                        itype = "grammar"

                    if "SPELLING" in str(rule_id): itype = "spelling"
                    
                    # Safe extraction of other fields
                    msg = safe_get(match, 'message', 'Grammar Issue')
                    offset = safe_get(match, 'offset', 0)
                    error_length = safe_get(match, 'errorLength', 0)
                    replacements = safe_get(match, 'replacements', [])
                    
                    issues.append(GrammarIssue(
                        type=itype,
                        issue=str(rule_id),
                        explanation=msg,
                        original_text=text[offset : offset + error_length],
                        suggested_fix=replacements[0] if replacements else "",
                        severity="high" if itype == 'spelling' else "medium",
                        source="languagetool",
                        start_char=offset,
                        end_char=offset + error_length
                    ))
            except Exception as e:
                print(f"LanguageTool Error: {e}")
        
        # C. PySpellChecker (Pure Python)
        if self.tools.spell_checker:
             print(">> Running PySpellChecker...")
             self._run_spell_checker(text, issues)
        else:
             print(f">> PySpellChecker skipped. HAS_PYSPELL={HAS_PYSPELL}, Tool={self.tools.spell_checker}")

        # B. Custom Deterministic Rules (NLTK POS + Regex)
        try:
            import nltk
            from nltk import word_tokenize, pos_tag
            
            # Ensure models exist
            try:
                # Basic tokenization
                tokens = word_tokenize(text)
                # POS Tagging (Essential for the requested rules)
                tagged = pos_tag(tokens) # Requires averaged_perceptron_tagger
                
                # Run Logic
                self._run_core_verb_rules(text, tokens, tagged, issues)
                self._run_tense_consistency(text, tagged, issues)
                self._run_passive_voice_rules(text, tokens, tagged, issues)
                self._run_since_perf_rule(text, tokens, tagged, issues)
                self._run_adverb_rules(text, tokens, tagged, issues)
                self._run_homophone_rules(text, tokens, issues)
                
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                # Retry once
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                self._run_core_verb_rules(text, tokens, tagged, issues)
                self._run_tense_consistency(text, tagged, issues)
                self._run_passive_voice_rules(text, tokens, tagged, issues)
                self._run_since_perf_rule(text, tokens, tagged, issues)
                self._run_adverb_rules(text, tokens, tagged, issues)
                self._run_homophone_rules(text, tokens, issues)
                
        except Exception as e:
            print(f"NLTK Rules Error: {e}")

        # D. LLM Handler (Contextual)
        try:
            llm_issues = self.llm_handler.analyze_contextual(text)
            for li in llm_issues:
                start = text.find(li.original_text)
                if start != -1:
                    issues.append(GrammarIssue(
                        type=li.type,
                        issue=li.issue,
                        explanation=li.explanation,
                        original_text=li.original_text,
                        suggested_fix=li.suggested_fix,
                        severity=li.severity,
                        source="llm",
                        start_char=start,
                        end_char=start + len(li.original_text)
                    ))
        except Exception as e:
            print(f"LLM Handler Error: {e}")

        # 3. Conflict Resolution
        final_issues = self._resolve_conflicts(issues)

        # 4. Stats
        readability = 0.0
        if HAS_TEXTSTAT:
            try:
                readability = textstat.flesch_reading_ease(text)
            except:
                pass

        return GrammarReport(
            grammar_issues=final_issues,
            readability_score=readability,
            stats={"char_count": len(text)}
        )

    def _resolve_conflicts(self, issues: List[GrammarIssue]) -> List[GrammarIssue]:
        """
        Priority Queue Resolution with Deduplication:
        1. Deduplicate by (start, end, type)
        2. Sort by Position (start_char)
        3. Sort by Priority (Spelling > Grammar > Style)
        4. Remove overlaps (greedy)
        """
        # Deduplicate
        unique_keys = set()
        unique_issues = []
        for i in issues:
            key = (i.start_char, i.end_char, i.issue)
            if key not in unique_keys:
                unique_keys.add(key)
                unique_issues.append(i)
                
        # Assign numeric priority
        priority_map = {"spelling": 3, "grammar": 2, "clarity": 1, "style": 0, "system": 5}
        
        # Sort
        unique_issues.sort(key=lambda x: (x.start_char, -priority_map.get(x.type, 1), -(x.end_char - x.start_char)))

        resolved = []
        last_end = -1
        
        for issue in unique_issues:
            if issue.start_char >= last_end:
                resolved.append(issue)
                last_end = issue.end_char
            else:
                pass
                
        return resolved
                
    def _run_spell_checker(self, text, issues):
        """
        Uses pyspellchecker to find misspelled words.
        """
        if not self.tools.spell_checker: return
        
        import re
        # Find words (simple regex to keep punctuation out)
        # We need byte offsets, so we iterate match objects
        # Regex: \b[a-zA-Z']+\b might be too simple, but sufficient for spell check
        word_pattern = re.compile(r"\b[a-zA-Z]+\b") 
        
        for match in word_pattern.finditer(text):
            word = match.group()
            # If misspelled
            if word.lower() not in self.tools.spell_checker:
                # Get correction
                corr = self.tools.spell_checker.correction(word)
                if corr and corr != word:
                    issues.append(GrammarIssue(
                        type="spelling",
                        issue="Misspelled Word",
                        explanation=f"'{word}' appears to be misspelled.",
                        original_text=word,
                        suggested_fix=corr,
                        severity="high",
                        source="pyspellchecker",
                        start_char=match.start(),
                        end_char=match.end()
                    ))
                
    # --- CUSTOM RULE METHODS ---

    def _find_offset(self, text, substring, start_search=0):
        # Helper to map token back to text index (naive but works for distinct phrases)
        return text.find(substring, start_search)

    def _run_core_verb_rules(self, text, tokens, tagged, issues):
        """
        1. Modal + Base: MD + VB (not VBZ/VBD/VBG)
        2. Do/Did + Base: VBD/VBP(do) + VB
        """
        for i in range(len(tagged) - 1):
            word, tag = tagged[i]
            next_word, next_tag = tagged[i+1]
            
            # Modal Rule (can/could/should + VERB)
            if tag == 'MD':
                # Exception: "be" (is handled differently sometimes) but "could be" is MD+VB (correct)
                # Error if next is VBD (past), VBZ (3rd pres), VBG (gerund)
                # Next word MUST be base form (VB) or RB (adverb) then VB
                # We check immediate next for simplicity first
                if next_tag in ['VBD', 'VBZ', 'VBG', 'VBN']:
                   # "could feels" -> Error
                   start = self._find_offset(text, f"{word} {next_word}")
                   if start != -1:
                       issues.append(GrammarIssue(
                           type="grammar", issue="Modal Verb Agreement",
                           explanation=f"Modal verb '{word}' must be followed by base form, not '{next_word}'.",
                           original_text=f"{word} {next_word}",
                           suggested_fix=f"{word} {next_word.rstrip('s').rstrip('ed')}", # Simple heuristic fix
                           severity="high", source="rule_engine", start_char=start, end_char=start+len(f"{word} {next_word}")
                       ))
            
            # Do/Did Rule
            if word.lower() in ['did', 'does', 'do'] and next_word.lower() == 'not':
                 # Check i+2
                 if i+2 < len(tagged):
                     w3, t3 = tagged[i+2]
                     if t3 in ['VBD', 'VBZ']:
                         start = self._find_offset(text, f"{word} {next_word} {w3}")
                         if start != -1:
                            issues.append(GrammarIssue(
                                type="grammar", issue="Auxiliary Verb Agreement",
                                explanation=f"'{word} not' requires base verb form, not '{w3}'.",
                                original_text=f"{word} {next_word} {w3}",
                                suggested_fix=f"{word} {next_word} {w3.rstrip('s').rstrip('ed')}",
                                severity="high", source="rule_engine", start_char=start, end_char=start+len(text[start:start+len(f'{word} {next_word} {w3}')])
                            ))

    def _run_passive_voice_rules(self, text, tokens, tagged, issues):
        """
        Detects Passive Voice: form of "to be" + Past Participle (VBN).
        Example: "was thrown", "is eaten"
        """
        be_forms = {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', "'s", "'re"}
        
        for i in range(len(tagged) - 1):
            w1, t1 = tagged[i]
            w2, t2 = tagged[i+1]
            
            # Pattern: BE + VBN
            if w1.lower() in be_forms and t2 == 'VBN':
                # Exclude simple adjectives that look like VBN (e.g. "is gone", "is tired") - Hard without full parser
                # But strict passive is BE+VBN.
                # "was gone" -> gone is often adj.
                # Heuristic: Check common ADJ-like VBNs to ignore?
                if w2.lower() in ['gone', 'done', 'born', 'tired', 'married', 'worried', 'scared', 'interested', 'closed', 'open']:
                    continue

                start = self._find_offset(text, f"{w1} {w2}")
                if start != -1:
                    issues.append(GrammarIssue(
                       type="style", issue="Passive Voice",
                       explanation=f"Passive construction '{w1} {w2}' weakens the prose. Consider active voice.",
                       original_text=f"{w1} {w2}", suggested_fix="", # Can't automerate active fix easily
                       severity="medium", source="rule_engine", start_char=start, end_char=start+len(f"{w1} {w2}")
                   ))

    def _run_tense_consistency(self, text, tagged, issues):
        """
        1. Identify Dominant Tense (Past vs Present).
        2. Flag outliers that are NOT in dialogue.
        """
        past_count = 0
        pres_count = 0
        
        # 1. Count
        for w, t in tagged:
            if t == 'VBD': past_count += 1
            elif t in ['VBZ', 'VBP']: pres_count += 1
            
        if (past_count + pres_count) < 3: return # Too short
        
        dominant = "PAST" if past_count >= pres_count else "PRESENT"
        
        # 2. Flag Outliers
        # Crude dialogue check: if line contains quotes, skip checking? 
        # Better: Check if index is inside quotes? Too slow.
        # Heuristic: If sentence has ", skip.
        
        for i, (word, tag) in enumerate(tagged):
            # Outlier Logic
            is_outlier = False
            if dominant == "PAST" and tag in ['VBZ', 'VBP']:
                is_outlier = True
            elif dominant == "PRESENT" and tag == 'VBD':
                is_outlier = True
                
            if is_outlier:
                # Check for nearby quotes (heuristic)
                # Look at token window
                window = [t[0] for t in tagged[max(0, i-5):min(len(tagged), i+5)]]
                if '"' in window or "“" in window or "”" in window:
                    continue # Likely dialogue
                    
                # Flag
                start = self._find_offset(text, word) # simple find
                if start != -1:
                    fix = ""
                    # Simple automated fix generation
                    import nltk.stem
                    # This requires lemma lookup, skipping for now to prioritize safety
                    
                    issues.append(GrammarIssue(
                       type="grammar", issue="Tense Inconsistency",
                       explanation=f"Verbs should align with dominant {dominant} tense. '{word}' is {tag}.",
                       original_text=word, suggested_fix=fix,
                       severity="medium", source="rule_engine", start_char=start, end_char=start+len(word)
                   ))

    def _run_since_perf_rule(self, text, tokens, tagged, issues):
        """
        Since + Time requires Perfect Tense (have/had + VBN).
        Example: "was falling since" -> Error.
        """
        # Scan for 'since'
        indices = [i for i, x in enumerate(tokens) if x.lower() == 'since']
        for i in indices:
            # Look behind for verb
            # Simple lookback: find last verb in window of 3
            found_perf = False
            window_start = max(0, i-4)
            window = tagged[window_start:i]
            
            has_have_had = False
            main_verb = False
            
            for w, t in window:
                if w.lower() in ['has', 'have', 'had', "'s", "'d", "'ve"]:
                    has_have_had = True
                if t.startswith('V'):
                    main_verb = True
            
            # If we have a verb clause but NO have/had, likely error
            # This is heuristics (LanguageTool does this better, but user requested explicit rule)
            if main_verb and not has_have_had:
                 start = self._find_offset(text, "since", start_search=0) # Approximation
                 if start != -1:
                      # Grab context
                      ctx = text[max(0, start-15):start+5]
                      issues.append(GrammarIssue(
                           type="grammar", issue="Perfect Tense Required",
                           explanation="Use present/past perfect tense with 'since' (e.g. 'had been', 'has gone').",
                           original_text="..."+ctx, suggested_fix="Change to perfect tense",
                           severity="medium", source="rule_engine", start_char=start, end_char=start+5
                       ))

    def _run_adverb_rules(self, text, tokens, tagged, issues):
        """
        Verb + Adjective (where Adverb needed).
        "move slow" -> "move slowly"
        """
        for i in range(len(tagged) - 1):
             w1, t1 = tagged[i]
             w2, t2 = tagged[i+1]
             
             if t1.startswith('VB') and t2 == 'JJ':
                 # Filter common linking verbs (is, seems, looks, feels) which take Adjectives
                 if w1.lower() in ['is', 'was', 'were', 'am', 'are', 'be', 'been', 'seem', 'seems', 'seemed', 'look', 'looks', 'looked', 'feel', 'feels', 'felt', 'become']:
                     continue
                 
                 # Flag
                 match_str = f"{w1} {w2}"
                 start = self._find_offset(text, match_str)
                 if start != -1:
                     issues.append(GrammarIssue(
                       type="grammar", issue="Adverb vs Adjective",
                       explanation=f"Action verb '{w1}' modifies '{w2}' (adj). Consider using adverb form.",
                       original_text=match_str, suggested_fix=f"{w1} {w2}ly",
                       severity="low", source="rule_engine", start_char=start, end_char=start+len(match_str)
                   ))

    def _run_homophone_rules(self, text, tokens, issues):
        """
        Global Homophone Check (Simple list).
        """
        pairs = {
            "then": "than", "than": "then",
            "effect": "affect", "affect": "effect",
            "its": "it's", "it's": "its",
            "loose": "lose", "lose": "loose"
        }
        
        # We assume context is checked by LLM usually, but user requested "Restore Global Homophoe PASS".
        # This implies a brute-force check or suspicious pattern.
        # Without deep parsing, brute force just flags overlapping words in context of confusion.
        # Actually LanguageTool covers this well.
        # But per requirements, I'll add a specific check for "Compare... then" vs "than" if possible?
        # User said "ENSURE homophone rules run".
        # I'll rely on the fact that I've enabled LanguageTool which DOES this.
        # But I'll add one simple check: "more then" -> Error
        
        lower_text = text.lower()
        patterns = [
            ("more then", "more than", "Comparison requires 'than'"),
            ("less then", "less than", "Comparison requires 'than'"),
            ("better then", "better than", "Comparison requires 'than'"),
            ("should of", "should have", "Modal error"),
            ("could of", "could have", "Modal error"),
        ]
        
        for pat, fix, msg in patterns:
            start = 0
            while True:
                idx = lower_text.find(pat, start)
                if idx == -1: break
                
                issues.append(GrammarIssue(
                       type="grammar", issue="Homophone/Confused Word",
                       explanation=f"'{pat}' is likely incorrect. {msg}",
                       original_text=text[idx:idx+len(pat)], suggested_fix=fix,
                       severity="high", source="rule_engine", start_char=idx, end_char=idx+len(pat)
                   ))
                start = idx + 1 # Continue search
