# LLM-Powered Features

Plasticity is **provider-agnostic** — it never imports a model SDK. You supply one callable,
`fn(prompt) -> str` (or an object with `.complete(prompt)`), and the LLM-aware components light
up. Every one of them **falls back to its deterministic baseline** if the call fails or the
output can't be parsed, so nothing ever hard-depends on a model.

```python
def my_llm(prompt: str, **kwargs) -> str:
    ...  # call your model of choice
    return "the model's text"

agent = PlasticAgent(name="copilot", llm_callback=my_llm)
```

## What upgrades

| Component | Without LLM | With LLM |
| --- | --- | --- |
| **Reflection** | keyword/rule classification | model-authored lesson + type |
| **Contradiction** | negation/antonym/sentiment heuristic | semantic entailment score |
| **Reasoning market** | six deterministic critics | + an LLM critic in the auction |
| **Self-Refine** | rubric checks + notes | model critique-and-rewrite |

## Structured output

The framework asks the model for JSON and parses it robustly (it tolerates code fences and
surrounding prose). For example, the reflector expects:

```json
{"lesson_type": "reasoning", "content": "...", "confidence": 0.8, "tags": ["accuracy"]}
```

If the model returns prose or invalid JSON, the deterministic path is used instead — your agent
keeps working.

## Cost control

Semantic contradiction checking only calls the LLM for pairs that pass a cheap lexical
relatedness gate; unrelated memories short-circuit to a score of 0 with no model call.

## Using components directly

```python
from plasticity_agent import Reflector, ReasoningMarket, ContradictionChecker, SelfRefine

Reflector(llm=my_llm).create_lesson(...)
ReasoningMarket(llm=my_llm).deliberate("...")
ContradictionChecker(llm=my_llm).pair("a", "b")
SelfRefine(llm_callback=lambda prompt, rubric: my_llm(prompt)).refine("...")
```
