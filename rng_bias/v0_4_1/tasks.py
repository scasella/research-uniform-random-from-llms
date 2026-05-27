"""Five product-shaped diversity tasks, five scenarios each.

Each scenario asks the model to generate 20 distinct outputs. Scenarios vary
across domains within a task so we measure whether the v0.4 LoRA's entropy
restoration transfers to *useful* diversity, not just to one bug type or one
naming brief.

Prompts are hand-crafted for the pilot. A follow-up could swap in public-source
prompts (HumanEval functions for unit_tests, real Jira archives for agent_plans)
for external-validity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioPrompt:
    scenario_id: str
    system: str
    user: str


@dataclass(frozen=True)
class ProductTask:
    task_id: str
    label: str
    expected_output_kind: str
    n_outputs: int
    scenarios: tuple[ScenarioPrompt, ...]
    judge_rubric_key: str
    validity_fn_key: str


_BUG_SYS = (
    "You are an experienced staff engineer triaging a production bug. "
    "Given the bug report below, generate exactly 20 distinct hypotheses for the root cause. "
    "Number each hypothesis 1-20, one per line. Be specific; avoid restating the bug."
)

_BUG_SCENARIOS = (
    ScenarioPrompt(
        scenario_id="bug_h_01",
        system=_BUG_SYS,
        user=(
            "Bug: NullPointerException in `HashMap.put()` inside our order-processing pipeline. "
            "Stack trace points to a worker thread; only reproduces under high concurrent load. "
            "Started after we upgraded JDK 17 to 21."
        ),
    ),
    ScenarioPrompt(
        scenario_id="bug_h_02",
        system=_BUG_SYS,
        user=(
            "Bug: a single Python pytest fails ~5% of the time on GitHub Actions CI runs but "
            "has never failed on any developer's local machine across 30+ runs. The test "
            "asserts a JSON payload deep-equals a fixture."
        ),
    ),
    ScenarioPrompt(
        scenario_id="bug_h_03",
        system=_BUG_SYS,
        user=(
            "Bug: a React checkout button stops responding to clicks after roughly 3 minutes "
            "of an active session. No console errors. Page is still scrollable. A hard refresh "
            "fixes it. Only reported by Chrome users on macOS."
        ),
    ),
    ScenarioPrompt(
        scenario_id="bug_h_04",
        system=_BUG_SYS,
        user=(
            "Bug: a PostgreSQL query is running 50x slower in production than in staging, "
            "despite identical data, identical schemas, and matching `EXPLAIN ANALYZE` plans. "
            "Both clusters are RDS, same instance class, same Postgres version."
        ),
    ),
    ScenarioPrompt(
        scenario_id="bug_h_05",
        system=_BUG_SYS,
        user=(
            "Bug: a Docker container is OOM-killed within 90 seconds only when running in "
            "our Kubernetes cluster. Running the identical image under docker-compose on the "
            "same host stays under 500MB RSS indefinitely."
        ),
    ),
)


_TEST_SYS = (
    "You are an experienced test-writer for Python. "
    "Given the function signature and docstring below, generate exactly 20 distinct pytest unit tests. "
    "Each test must be a complete, runnable function starting with `def test_`. "
    "Cover edge cases, error paths, boundary values, and unusual inputs. Avoid duplicates."
)

_TEST_SCENARIOS = (
    ScenarioPrompt(
        scenario_id="test_01",
        system=_TEST_SYS,
        user=(
            "```python\n"
            "def chunk(items: list, size: int) -> list[list]:\n"
            "    \"\"\"Return items grouped into sub-lists of length `size`. "
            "Last group may be shorter. Raise ValueError if size <= 0.\"\"\"\n"
            "```"
        ),
    ),
    ScenarioPrompt(
        scenario_id="test_02",
        system=_TEST_SYS,
        user=(
            "```python\n"
            "def parse_duration(s: str) -> int:\n"
            "    \"\"\"Parse a duration string like '1h30m', '45s', '2d12h' into total seconds. "
            "Supports d/h/m/s suffixes. Raise ValueError on malformed input.\"\"\"\n"
            "```"
        ),
    ),
    ScenarioPrompt(
        scenario_id="test_03",
        system=_TEST_SYS,
        user=(
            "```python\n"
            "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
            "    \"\"\"Given a list of (start, end) intervals, return the minimal list of "
            "non-overlapping intervals covering the same range. Empty input returns [].\"\"\"\n"
            "```"
        ),
    ),
    ScenarioPrompt(
        scenario_id="test_04",
        system=_TEST_SYS,
        user=(
            "```python\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    \"\"\"Return True iff `s` is a palindrome, ignoring case, whitespace, "
            "and punctuation. Empty string is considered a palindrome.\"\"\"\n"
            "```"
        ),
    ),
    ScenarioPrompt(
        scenario_id="test_05",
        system=_TEST_SYS,
        user=(
            "```python\n"
            "def rate_limited(fn, max_per_sec: int):\n"
            "    \"\"\"Decorator: wrap `fn` so it can be called at most `max_per_sec` times "
            "per rolling second. Excess calls block until a slot is free.\"\"\"\n"
            "```"
        ),
    ),
)


_NAME_SYS = (
    "You are a brand-naming consultant. "
    "Given the product brief below, suggest exactly 20 distinct product names. "
    "One name per line, numbered 1-20. Names should be 1-4 words, easy to say, no domain suffixes, "
    "no obvious trademark conflicts. Aim for variety in tone and form across the 20."
)

_NAME_SCENARIOS = (
    ScenarioPrompt(
        scenario_id="name_01",
        system=_NAME_SYS,
        user=(
            "Brief: a privacy-first browser extension that blocks AI training scrapers from "
            "reading webpages while you browse. Audience: privacy-conscious developers and "
            "journalists. Tone: trustworthy, slightly technical, not playful."
        ),
    ),
    ScenarioPrompt(
        scenario_id="name_02",
        system=_NAME_SYS,
        user=(
            "Brief: an async stand-up tool for small distributed engineering teams. No "
            "scheduled meetings; the tool posts a thread in Slack each morning and aggregates "
            "responses. Audience: 5-15-person remote eng teams. Tone: friendly, anti-meeting."
        ),
    ),
    ScenarioPrompt(
        scenario_id="name_03",
        system=_NAME_SYS,
        user=(
            "Brief: a CLI that turns Postgres `EXPLAIN ANALYZE` output into an interactive "
            "flame graph in the terminal. Audience: backend engineers debugging slow queries. "
            "Tone: precise, developer-tool, can be one technical word."
        ),
    ),
    ScenarioPrompt(
        scenario_id="name_04",
        system=_NAME_SYS,
        user=(
            "Brief: a monthly subscription box that ships three small-batch craft cocktails "
            "(pre-mixed, just add ice) chosen by a guest bartender each month. Audience: "
            "home bar enthusiasts. Tone: warm, slightly upscale, lifestyle."
        ),
    ),
    ScenarioPrompt(
        scenario_id="name_05",
        system=_NAME_SYS,
        user=(
            "Brief: a mobile habit-tracking app that deliberately removes streaks and "
            "gamification. Tracks habits, shows long-term trends, no badges, no XP, no "
            "notifications nagging users. Audience: people burnt out on streak-based apps. "
            "Tone: calm, minimalist, anti-gamification."
        ),
    ),
)


_ARG_SYS = (
    "You are facilitating a structured debate on an internal company proposal. "
    "Given the proposal below, generate exactly 20 distinct stakeholder objections. "
    "Each objection should: name the stakeholder perspective (e.g., 'as a customer', 'as the "
    "security lead'), state a specific concern, and be coherent on its own. "
    "Number 1-20, one per line. Cover varied perspectives."
)

_ARG_SCENARIOS = (
    ScenarioPrompt(
        scenario_id="arg_01",
        system=_ARG_SYS,
        user=(
            "Proposal: split the team's monorepo into 12 separate repositories, one per "
            "service. Each repo would own its CI, dependencies, and release cadence."
        ),
    ),
    ScenarioPrompt(
        scenario_id="arg_02",
        system=_ARG_SYS,
        user=(
            "Proposal: replace human code review on all pull requests with AI-only code "
            "review. Humans would only review changes the AI flags as needing attention."
        ),
    ),
    ScenarioPrompt(
        scenario_id="arg_03",
        system=_ARG_SYS,
        user=(
            "Proposal: ban Slack DMs company-wide. All work conversations must happen in "
            "public channels. Only HR-sensitive discussions are exempt."
        ),
    ),
    ScenarioPrompt(
        scenario_id="arg_04",
        system=_ARG_SYS,
        user=(
            "Proposal: charge customers $5/month for access to human support. Free-tier "
            "customers would have only AI chatbot and documentation."
        ),
    ),
    ScenarioPrompt(
        scenario_id="arg_05",
        system=_ARG_SYS,
        user=(
            "Proposal: deprecate the desktop app within 6 months and migrate all users to "
            "the web version. The desktop app accounts for 35% of active users."
        ),
    ),
)


_PLAN_SYS = (
    "You are a senior engineer planning execution for a Jira ticket. "
    "Given the ticket below, write exactly 20 distinct multi-step plans for how you'd address it. "
    "Each plan should be a numbered list of 3-7 concrete steps. Separate the 20 plans with "
    "'--- PLAN N ---' headers. Aim for variety in approach, not just step ordering."
)

_PLAN_SCENARIOS = (
    ScenarioPrompt(
        scenario_id="plan_01",
        system=_PLAN_SYS,
        user=(
            "Ticket: 'Investigate why our payment webhook is missing roughly 0.3% of "
            "Stripe events. Affected customers report orders not marked paid.'"
        ),
    ),
    ScenarioPrompt(
        scenario_id="plan_02",
        system=_PLAN_SYS,
        user=(
            "Ticket: 'Add SSO via Okta for enterprise customers. Must support SAML 2.0 and "
            "SCIM provisioning. Deadline: end of Q3. Owner: backend team.'"
        ),
    ),
    ScenarioPrompt(
        scenario_id="plan_03",
        system=_PLAN_SYS,
        user=(
            "Ticket: 'Customer escalation: a Pro-tier customer reports their workspace data "
            "was deleted after they were the only owner and their account was suspended for "
            "non-payment. Recover the data or produce an explanation.'"
        ),
    ),
    ScenarioPrompt(
        scenario_id="plan_04",
        system=_PLAN_SYS,
        user=(
            "Ticket: 'Reduce p99 latency on /api/search from 800ms to under 200ms within "
            "one quarter. Endpoint serves 10M requests/day; existing implementation uses "
            "Postgres full-text search.'"
        ),
    ),
    ScenarioPrompt(
        scenario_id="plan_05",
        system=_PLAN_SYS,
        user=(
            "Ticket: 'Ship a self-serve account-deletion feature compliant with GDPR "
            "Article 17. Must delete all PII and content within 30 days, with audit logs.'"
        ),
    ),
)


TASKS: tuple[ProductTask, ...] = (
    ProductTask(
        task_id="bug_hypotheses",
        label="Bug hypotheses",
        expected_output_kind="numbered_list",
        n_outputs=20,
        scenarios=_BUG_SCENARIOS,
        judge_rubric_key="bug_hypotheses",
        validity_fn_key="bug_hypotheses",
    ),
    ProductTask(
        task_id="unit_tests",
        label="Unit tests",
        expected_output_kind="code_block",
        n_outputs=20,
        scenarios=_TEST_SCENARIOS,
        judge_rubric_key="unit_tests",
        validity_fn_key="unit_tests",
    ),
    ProductTask(
        task_id="product_names",
        label="Product names",
        expected_output_kind="numbered_list",
        n_outputs=20,
        scenarios=_NAME_SCENARIOS,
        judge_rubric_key="product_names",
        validity_fn_key="product_names",
    ),
    ProductTask(
        task_id="stakeholder_arguments",
        label="Stakeholder arguments",
        expected_output_kind="numbered_list",
        n_outputs=20,
        scenarios=_ARG_SCENARIOS,
        judge_rubric_key="stakeholder_arguments",
        validity_fn_key="stakeholder_arguments",
    ),
    ProductTask(
        task_id="agent_plans",
        label="Agent plans",
        expected_output_kind="plans_block",
        n_outputs=20,
        scenarios=_PLAN_SCENARIOS,
        judge_rubric_key="agent_plans",
        validity_fn_key="agent_plans",
    ),
)


TASK_BY_ID: dict[str, ProductTask] = {task.task_id: task for task in TASKS}


def get_task(task_id: str) -> ProductTask:
    if task_id not in TASK_BY_ID:
        raise KeyError(f"Unknown v0.4.1 task id: {task_id!r}")
    return TASK_BY_ID[task_id]


__all__ = ["ScenarioPrompt", "ProductTask", "TASKS", "TASK_BY_ID", "get_task"]
