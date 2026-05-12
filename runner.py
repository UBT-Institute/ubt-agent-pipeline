from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = ROOT / "config"
DEFAULT_INSTRUCTIONS_DIR = ROOT / "instructions"
DEFAULT_RUNS_DIR = ROOT / "runs"


@dataclass
class StepResult:
    step_id: str
    agent_id: str
    prompt: str
    output: str


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_agents(path: Path) -> dict[str, dict[str, Any]]:
    data = read_yaml(path)
    return {str(item["id"]): item for item in data}


def load_workflow(path: Path, workflow_id: str) -> dict[str, Any]:
    data = read_yaml(path)
    for workflow in data:
        if str(workflow["id"]) == workflow_id:
            return workflow
    raise KeyError(f"Workflow '{workflow_id}' not found in {path}")


def run_command(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def build_repo_map(repo_path: Path, limit: int = 250) -> str:
    tracked = run_command(["git", "ls-files"], cwd=repo_path)
    if not tracked:
        return ""
    lines = tracked.splitlines()
    return "\n".join(lines[:limit])


def collect_search_hits(repo_path: Path, query: str, limit: int = 20) -> str:
    normalized = query.strip()
    if not normalized:
        return ""

    patterns = [token for token in re.split(r"\s+", normalized) if len(token) >= 4][:4]
    if not patterns:
        return ""

    search_expr = "|".join(re.escape(token) for token in patterns)
    commands = [
        ["git", "grep", "-nE", search_expr, "--", "."],
        ["grep", "-RInE", search_expr, "."],
    ]

    for command in commands:
        output = run_command(command, cwd=repo_path)
        if output:
            return "\n".join(output.splitlines()[:limit])
    return ""


def read_instruction(instructions_dir: Path, instruction_name: str | None) -> str:
    if not instruction_name:
        return ""
    path = instructions_dir / instruction_name
    if not path.exists():
        raise FileNotFoundError(f"Instruction file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def compose_prompt(
    task: str,
    repo_path: Path,
    repo_map: str,
    search_hits: str,
    previous_outputs: list[StepResult],
    step: dict[str, Any],
    instruction_text: str,
) -> str:
    previous = "\n\n".join(
        f"[{item.step_id} :: {item.agent_id}]\n{item.output}" for item in previous_outputs
    )
    step_prompt = str(step.get("prompt") or "Produce the requested output for this workflow step.")
    parts = [
        "TASK",
        task,
        "",
        "STEP",
        f"ID: {step['id']}",
        f"NAME: {step.get('name', step['id'])}",
        step_prompt,
        "",
        "INSTRUCTIONS",
        instruction_text or "No extra instructions.",
        "",
        "REPOSITORY MAP",
        repo_map or "No repository map available.",
        "",
        "SEARCH HITS",
        search_hits or "No direct search hits.",
        "",
        "PREVIOUS OUTPUTS",
        previous or "No previous outputs.",
        "",
        "OUTPUT REQUIREMENTS",
        "Be explicit about assumptions, cite file paths when relevant, and keep mathematical claims auditable.",
    ]
    return "\n".join(parts).strip()


class EchoClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"[echo]\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"


class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str):
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        self.client = Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()


class GeminiClient:
    def __init__(self, model: str):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return getattr(response, "text", "") or ""


def make_client(agent_cfg: dict[str, Any]):
    provider = str(agent_cfg.get("provider", "echo")).lower()
    model = str(agent_cfg.get("model") or "")
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "gemini":
        return GeminiClient(model)
    return EchoClient()


def write_run_artifacts(run_dir: Path, task: str, workflow: dict[str, Any], results: list[StepResult]) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.md"
    json_path = run_dir / "results.json"

    summary_lines = [
        f"# {workflow.get('name', workflow['id'])}",
        "",
        f"Task: {task}",
        "",
    ]
    for result in results:
        summary_lines.extend(
            [
                f"## {result.step_id} ({result.agent_id})",
                "",
                result.output.strip() or "No output.",
                "",
            ]
        )

    summary_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "task": task,
                "workflow_id": workflow["id"],
                "results": [result.__dict__ for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path, json_path


def execute_workflow(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo).resolve()
    agents = load_agents(Path(args.agents).resolve())
    workflow = load_workflow(Path(args.workflows).resolve(), args.workflow)
    repo_map = build_repo_map(repo_path)
    search_hits = collect_search_hits(repo_path, args.task)
    results: list[StepResult] = []

    for step in workflow["steps"]:
        agent_id = str(step["agent"])
        agent_cfg = agents[agent_id]
        instruction_name = agent_cfg.get("instruction")
        system_prompt = read_instruction(Path(args.instructions).resolve(), instruction_name)
        prompt = compose_prompt(args.task, repo_path, repo_map, search_hits, results, step, system_prompt)
        client = make_client(agent_cfg)
        output = client.generate(system_prompt=system_prompt, user_prompt=prompt)
        results.append(StepResult(step_id=str(step["id"]), agent_id=agent_id, prompt=prompt, output=output))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir).resolve() / f"{timestamp}-{workflow['id']}"
    summary_path, json_path = write_run_artifacts(run_dir, args.task, workflow, results)
    print(f"summary_path={summary_path}")
    print(f"results_path={json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UBT GitHub-native research pipeline")
    parser.add_argument("--task", required=True, help="Research task description")
    parser.add_argument("--workflow", default="ubt_derivation_review", help="Workflow identifier")
    parser.add_argument("--repo", default=".", help="Repository path to analyze")
    parser.add_argument("--agents", default=str(DEFAULT_CONFIG_DIR / "agents.yaml"))
    parser.add_argument("--workflows", default=str(DEFAULT_CONFIG_DIR / "workflows.yaml"))
    parser.add_argument("--instructions", default=str(DEFAULT_INSTRUCTIONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RUNS_DIR))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return execute_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
