# Facet-conditioned prompt design

> **Context:** This section assumes familiarity with the taxonomy described in **Sections 1–3**.  
> See: [Taxonomy (Sections 1–3)](../taxonomy/taxonomy-guided-prompting.md).


We generate activities with a two-part prompt: a *fixed block* (always the same rules), and a facet-specific variable block (one facet at a time, carrying task, palettes, constraints, and few-shot examples). The fixed block instructs the model to produce short, kid-friendly Dutch activity sentences (CEFR A2–B1), exactly 16 lines, lowercase, no punctuation, one action per line, and to adhere to safety and formatting constraints. The *variable block* injects context (domain, subdomain, facet), a task, lexical palettes (suggested verbs/objects/settings), quotas/constraints, and three tone-setting examples.

> Note on language:  
> The prompt text below is documented in English for readability, but it is formulated in Dutch in the original and explicitly asks the model to output Dutch.

---


## Fixed block — **Pattern**

```text
You are an educational copywriter. Write SHORT, CHILD-FRIENDLY Dutch activity sentences that each express exactly one concrete action.

RULES
- Output Dutch only; no emoji; no English words
- Exactly 16 lines; each line numbered 1. … 16.
- Each sentence is 5–12 tokens; all lowercase
- No punctuation (no dots, commas, hyphens, colons, etc.)
- No intro or outro; output the sentences only
- Use the setting only as inspiration; do not name the location explicitly
- Vary verbs, objects, and settings; avoid duplicates
- Keep content safe and age-appropriate: no brands, no PII, no adult themes, no illegal or dangerous instructions
- No negation flips relative to examples; no exact quantities/money unless requested; no compound sentences

FORMAT
Give 16 lines, exactly numbered as:
1. <sentence>
2. <sentence>
...
16. <sentence>
```

---


## Variable block — **Pattern**

Each run targets one facet. We pass the facet context, task, palettes, constraints, and three examples.

```text
ASPECT: <Domain> → <Subdomain>
SUBASPECT/FACET: <Facet name>
TASK: <What to produce>

LEXICAL PALETTES (inspiration, optional)
- verbs: <comma-separated Dutch verbs>
- objects: <comma-separated Dutch objects/materials>
- settings: <comma-separated Dutch settings>

CONSTRAINTS (soft quotas)
- <short quota statements in English; optional>

EXAMPLES (Dutch, tone-setting; not seeds)
- <example 1>
- <example 2>
- <example 3>
```

**Example of a variable block for facet 1 of subdomain 1 of domain 1: Classroom instruction (facet_id: 1.1.1)**

```json
{
  "domain": "Education domain",
  "domain_id": "1",
  "subdomain": "Learning activities",
  "subdomain_id": "1.1",
  "facet": "Classroom instruction",
  "facet_id": "1.1.1",
  "task": "Formulate 16 activities where students take part in instruction-led teaching by listening, taking notes, asking questions, or taking turns.",
  "lexical": {
    "verbs": ["listen", "take notes", "write down", "ask", "answer", "take part", "respond", "summarize", "repeat", "clarify", "pay attention", "keep track", "highlight", "write along"],
    "objects": ["explanation", "instruction", "story", "presentation", "teacher's explanation", "clarification", "example", "demonstration", "notes", "diagram", "keywords", "main points"],
    "settings": ["classroom", "computer lab", "library corner", "gym"]
  },
  "constraints": {
    "verb_distribution": "at least 5 lines with listen/pay attention, 4 lines with take notes/write down, 4 lines with ask/answer, 3 mixed",
    "interaction": "at least 6 lines with active response/communication",
    "cognitive_levels": "vary between passive reception (listening) and active processing (asking questions, summarizing)"
  },
  "examples": [
    "listen attentively to the explanation",
    "ask a question about the subject matter",
    "write down the lesson's keywords"
  ]
}
```

The full series of variable blocks (facet 1.1.1 → facet 9.3.5) can be consulted in the Dutch original under config/prompt_variable_blocks/ (one JSON per subdomain: vb_1.1.json, vb_1.2.json, …): ../../config/prompt_variable_blocks/.