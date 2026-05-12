# UBT Agent Pipeline

GitHub-native AI research pipeline for UBT.

The design goal is to stay cheap, auditable, and serverless:

- orchestration runs in GitHub Actions
- model credentials stay in GitHub Secrets
- repository retrieval uses deterministic text search instead of a live RAG server
- agent behavior is driven by small instruction files and YAML workflow definitions

## Why no RAG yet

The current priority is reliability and low operating cost. Semantic indexing can help later, but for now the pipeline uses:

- `git ls-files` to build a repository map
- `git grep` with a plain `grep` fallback to retrieve nearby code and notes
- workflow outputs persisted as artifacts for human audit

This avoids standing up a vector database or background indexing service.

## Current team

- `planner`: decomposes a research task into auditable steps
- `derivation`: drafts symbolic derivations and marks gaps explicitly
- `verifier`: checks consistency and unsupported inference jumps
- `critic`: pressure-tests the argument and looks for cheap falsifiers
- `auditor`: prepares the final packet for principal researcher review

Agent definitions live in [ubt-agent-pipeline/config/agents.yaml](ubt-agent-pipeline/config/agents.yaml).
Workflow definitions live in [ubt-agent-pipeline/config/workflows.yaml](ubt-agent-pipeline/config/workflows.yaml).
Role instructions live in [ubt-agent-pipeline/instructions/planner.instructions.md](ubt-agent-pipeline/instructions/planner.instructions.md) and neighboring files.

## GitHub Actions flow

Workflow file: [UBT-Institute/.github/workflows/ubt-research-pipeline.yml](.github/workflows/ubt-research-pipeline.yml)

Supported triggers:

- manual `workflow_dispatch`
- issue labeled `research-task`

Outputs:

- `summary.md` artifact
- `results.json` artifact
- optional issue comment with a summary excerpt

## Required secrets

Configure only the providers you plan to use:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

If a configured agent uses a provider whose secret is missing, the run will fail fast.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ubt-agent-pipeline/requirements.txt
python ubt-agent-pipeline/runner.py \
	--task "Check whether the current draft defines the projection operator consistently." \
	--workflow ubt_derivation_review \
	--repo .
```

Artifacts are written into `ubt-agent-pipeline/runs/`.

## Next practical steps

1. Point the pipeline at the main theory repository once the workflow shape is stable.
2. Add a second workflow for bibliography and literature checking.
3. Introduce lightweight semantic indexing only if `git grep` stops being sufficient.
