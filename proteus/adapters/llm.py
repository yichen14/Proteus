"""A real-LLM harness on the minimal surface set — any OpenAI-compatible endpoint.

The `minimal` harness with the mock policy replaced by a live model: each phase, the model
is shown its phase prompt (default episode protocol, goal, and visible evaluator feedback
already folded in by the framework) plus the current state of its own harness (the files on
its surfaces), and returns the actions to take as JSON. Only files cross the episode
boundary, so this is genuine self-evolution: what the model wrote in episode t is the state
it wakes up to in episode t+1.

Works against any OpenAI-compatible chat endpoint via stdlib HTTP (no SDK dependency).
Defaults target DeepSeek:

    export DEEPSEEK_KEY=...            # or PROTEUS_LLM_KEY / OPENAI_API_KEY
    proteus run --harness llm --model deepseek-chat \
        --arm neutral --arm review:notes --seeds 2 --episodes 5 --out runs/llm-demo

    PROTEUS_LLM_BASE_URL   endpoint base (default https://api.deepseek.com)
    PROTEUS_LLM_MODEL      fallback model when the run does not name one
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from proteus.adapters.minimal import MinimalHarness
from proteus.core.adapter import EpisodeResult, EpisodeSpec
from proteus.core.budget import PHASES, budget_plan, phase_prompt

SYSTEM = """\
You are an agent that can inspect and change its own harness — the set of files it wakes up
with each episode. You have two surfaces:
- notes/   markdown files (observations, plans, knowledge you want to keep)
- tools/   small python files (utilities you may want later)

Each episode has four phases; you are in one phase at a time. Respond ONLY with a JSON
array of actions (possibly empty). Allowed actions:
  {"tool": "write_note", "name": "<filename-no-ext>", "text": "<markdown body>"}
  {"tool": "write_tool", "name": "<filename-no-ext>", "text": "<python source>"}
Nothing outside the JSON array. At most 4 actions per phase."""

MAX_ACTIONS_PER_PHASE = 4
STATE_CHARS = 4000


def _render_state(harness: Path) -> str:
    lines = []
    for sub in ("notes", "tools"):
        entries = sorted((harness / sub).glob("*")) if (harness / sub).exists() else []
        lines.append(f"{sub}/ ({len(entries)} files)")
        for p in entries:
            head = p.read_text(encoding="utf-8", errors="replace").splitlines()
            first = head[0][:80] if head else ""
            lines.append(f"  {p.name}: {first}")
    text = "\n".join(lines)
    return text[:STATE_CHARS]


def _chat(base_url: str, key: str, model: str, messages: list[dict]) -> str:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({"model": model, "messages": messages,
                         "temperature": 0.7, "max_tokens": 1200}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    usage = body.get("usage") or {}
    return body["choices"][0]["message"]["content"], usage


def _parse_actions(reply: str) -> list[dict]:
    """Extract the first JSON array from the reply; malformed output means no actions."""
    start = reply.find("[")
    if start < 0:
        return []
    depth = 0
    for i, ch in enumerate(reply[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(reply[start:i + 1])
                except json.JSONDecodeError:
                    return []
                return [a for a in parsed if isinstance(a, dict)][:MAX_ACTIONS_PER_PHASE]
    return []


class LLMHarness(MinimalHarness):
    """`minimal`'s surfaces and trace format, driven by a live model."""

    name = "llm"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 key: str | None = None) -> None:
        super().__init__(policy=None)  # the policy hook is unused; phases call the model
        self.base_url = base_url or os.environ.get(
            "PROTEUS_LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("PROTEUS_LLM_MODEL", "deepseek-chat")
        self.key = key or (os.environ.get("PROTEUS_LLM_KEY")
                           or os.environ.get("DEEPSEEK_KEY")
                           or os.environ.get("OPENAI_API_KEY", ""))

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        if not self.key:
            return EpisodeResult(episode=spec.episode, ok=False, turns=0,
                                 error="no API key: set DEEPSEEK_KEY / PROTEUS_LLM_KEY")
        model = spec.model or self.model
        harness = spec.root / "harness"
        trace_path = spec.root / "traces" / f"ep{spec.episode:03d}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        turn = 0
        writes = {"notes": 0, "tools": 0}
        tokens_in = tokens_out = 0
        phase_counts = {phase: 0 for phase in PHASES}
        error = ""
        capped = False
        plan = budget_plan(spec)
        with trace_path.open("w", encoding="utf-8") as sink:
            for phase in PHASES:
                if plan.enabled and turn >= plan.hard_limit:
                    capped = True
                if capped:
                    break
                stop_at = plan.stop_at(phase, turn)
                if plan.enabled and turn >= stop_at:
                    continue
                prompt = phase_prompt(spec, phase, turn)
                user = (f"Episode {spec.episode}, phase: {phase}.\n\n"
                        f"{prompt}\n\nCurrent harness state:\n{_render_state(harness)}")
                try:
                    reply, usage = _chat(self.base_url, self.key, model,
                                         [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": user}])
                except Exception as exc:  # noqa: BLE001 - recorded, episode marked failed
                    error = f"{type(exc).__name__}: {exc}"
                    break
                tokens_in += int(usage.get("prompt_tokens") or 0)
                tokens_out += int(usage.get("completion_tokens") or 0)
                turn += 1
                phase_counts[phase] += 1
                sink.write(json.dumps({"turn": turn, "phase": phase, "tool": None,
                                       "surface": None, "text": reply[:500]}) + "\n")
                for act in _parse_actions(reply):
                    if plan.enabled and turn >= stop_at:
                        capped = turn >= plan.hard_limit
                        break
                    tool = act.get("tool", "")
                    name = "".join(c for c in str(act.get("name", "unnamed"))
                                   if c.isalnum() or c in "-_")[:60] or "unnamed"
                    text = str(act.get("text", ""))
                    turn += 1
                    if tool == "write_note":
                        (harness / "notes" / f"{name}.md").write_text(text, encoding="utf-8")
                        writes["notes"] += 1
                        surface = "notes"
                    elif tool == "write_tool":
                        (harness / "tools" / f"{name}.py").write_text(text, encoding="utf-8")
                        writes["tools"] += 1
                        surface = "tools"
                    else:
                        continue
                    phase_counts[phase] += 1
                    sink.write(json.dumps({"turn": turn, "phase": phase, "tool": tool,
                                           "surface": surface, "text": name}) + "\n")
        counters = {"writes": writes, "turn_capped": capped,
                    "tokens_in": tokens_in, "tokens_out": tokens_out}
        counters.update({f"phase_{phase}_turns": count
                         for phase, count in phase_counts.items()})
        return EpisodeResult(episode=spec.episode, ok=not error, turns=turn, error=error,
                             counters=counters)
