# PH-FT Calculator Skill

## What this is

A portable agent skill that helps decide API-vs-self-host LLM economics and
fine-tuning ROI from any context the user provides — code, PRDs, traffic logs,
billing screenshots. It works in Claude Code, Cursor, Codex, or any harness
with a shell-execution tool and a web-fetch tool. The LLM gathers context,
fetches live pricing/quality data from the web, and calls a tiny deterministic
Python script in this folder for the math.

## Install

Symlink this `skill/` folder into your harness's skills directory. The skill
is self-contained, so a symlink keeps it in sync with `git pull`.

### Claude Code

```bash
ln -s "$PWD/skill" ~/.claude/skills/ph-ft
```

### Cursor

```bash
ln -s "$PWD/skill" ~/.cursor/skills/ph-ft
```

### Any other harness

Copy or symlink the `skill/` folder anywhere the harness reads skills from.
The skill is self-contained — no global config, no environment variables,
no package installs.

## Requirements

- Python 3.10+ (standard library only — no `pip install` needed).
- The harness must have a shell-execution tool and a web-fetch tool. Every
  modern agent harness (Claude Code, Cursor, Codex, etc.) does.

## How it works (one paragraph)

The LLM reads `SKILL.md`, loops with the user to gather inputs (current API
spend, traffic shape, quality bar, fine-tune intent), fetches live GPU prices
from vendor pages and live API/quality data from `models.dev` and
`lmarena.ai`, calls `python3 skill/calc.py inference` or
`python3 skill/calc.py finetune` with a JSON payload on stdin to do the
arithmetic, then writes a markdown report with a recommendation, a
comparison matrix, full derivations, and cited sources.

## Files

- `SKILL.md` — the agentic loop instructions the LLM follows
- `calc.py` — deterministic math (Python stdlib only)
- `references/ASSUMPTIONS.md` — math assumptions and calibration anchors
- `references/INPUTS.md` — input field contract and sensible defaults
- `references/GPU_SPECS.md` — static GPU physical specs (VRAM, BF16 TFLOPS)

## Updating

`git pull` in this repo. The skill folder is just files — no build step,
no install step.
