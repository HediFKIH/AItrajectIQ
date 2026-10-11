import asyncio

import pytest

from evaluation import runner
from services import prompt_advisor

SYSTEM_ITEM = {
    "id": "system", "target": "system_prompt", "title": "Prompt système recommandé (gpt-4.1)", "file": None,
    "current": "Tu es l'assistant du support.", "content": "Tu es l'assistant du support.\n\n## Consignes\n- Cite tes sources.",
    "changes": ["Réponses peu fidèles aux sources"], "model": "gpt-4.1", "calls": 12, "source": "rules",
}
CURSOR_ITEM = {
    "id": "cursor", "target": "cursor", "title": "Règle Cursor recommandée", "file": ".cursor/rules/aitrajectiq.mdc",
    "current": None, "content": "---\n...\n---\n\n- Fais ce qui est demandé.\n", "changes": [], "model": None,
    "calls": None, "source": "rules",
}


@pytest.fixture
def judge(monkeypatch):
    """A fake evaluation LLM that records its calls."""
    calls = []
    answers = []

    async def complete_json(system, user, max_tokens=500):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return answers.pop(0)

    monkeypatch.setattr(runner, "judge_tag", lambda: "fake@v4")
    monkeypatch.setattr(runner, "complete_json", complete_json)
    monkeypatch.setattr(prompt_advisor, "_cache", {})
    return calls, answers


def test_system_prompt_rewrite_is_cached(judge):
    calls, answers = judge
    answers.append({"prompt": "Tu es l'assistant du support. Cite la source de chaque affirmation.",
                    "changes": ["Ajoute la citation des sources"]})
    findings = ["Réponses peu fidèles aux sources : 50 %"]

    result = asyncio.run(prompt_advisor.rewrite(SYSTEM_ITEM, findings))
    again = asyncio.run(prompt_advisor.rewrite(SYSTEM_ITEM, findings))

    assert result["source"] == "ai" and result["id"] == "system"
    assert result["content"].endswith("Cite la source de chaque affirmation.")
    assert result["changes"] == ["Ajoute la citation des sources"]
    assert again == result and len(calls) == 1
    assert "Tu es l'assistant du support." in calls[0]["user"] and findings[0] in calls[0]["user"]


def test_cursor_rules_are_wrapped_in_a_rule_file(judge):
    calls, answers = judge
    answers.append({"rules": ["Ne modifie que les fichiers cités dans la demande.", ""], "changes": []})

    result = asyncio.run(prompt_advisor.rewrite(CURSOR_ITEM, ["Consignes mal suivies : 50 %"]))

    assert result["content"].startswith("---\n") and "alwaysApply: true" in result["content"]
    assert "- Ne modifie que les fichiers cités dans la demande.\n" in result["content"]
    assert "Cursor" in calls[0]["system"]


def test_rewrite_unavailable_without_a_judge(monkeypatch):
    monkeypatch.setattr(runner, "judge_tag", lambda: None)
    monkeypatch.setattr(prompt_advisor, "_cache", {})
    with pytest.raises(prompt_advisor.RewriteUnavailable):
        asyncio.run(prompt_advisor.rewrite(SYSTEM_ITEM, []))


def test_empty_answer_is_an_error(judge):
    _, answers = judge
    answers.append({"prompt": "  "})
    with pytest.raises(prompt_advisor.RewriteUnavailable):
        asyncio.run(prompt_advisor.rewrite(SYSTEM_ITEM, []))
