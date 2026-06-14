# Contributing to AIgnition

## Getting Started

```bash
git clone https://github.com/your-org/aignition.git
cd aignition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

All PRs must pass the full test suite:

```bash
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Code Style

- Follow existing conventions (no external formatter enforced)
- Type hints on all public functions
- No inline comments explaining *what* — only *why* (non-obvious constraints, workarounds)
- No docstrings longer than one sentence for simple functions

## Pull Request Guidelines

1. Branch from `main`
2. One logical change per PR
3. All tests must pass
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Do not commit `models/`, `dataset/*.parquet`, `dataset/*.pkl`, or `.env`

## Reporting Bugs

Open a GitHub Issue with:
- Python version
- Exact error message and traceback
- Minimal reproduction steps

## Directory Map

```
src/data/           CSV loaders and data harmonizer
src/features/       Feature store builder
src/models/         LightGBM training, cross-validation, autoregressive inference
src/simulation/     Hill curves, SLSQP optimizer, scenario simulation
src/copilot/        Insight engine, risk detector, recommender, LLM client
streamlit_app/      Multi-page Streamlit dashboard
tests/              Unit tests (mirror src/ structure)
```
