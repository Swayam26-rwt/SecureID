# Contributing to SecureID

Thank you for your interest in SecureID. This document explains the standards, workflow, and constraints all contributors must follow.

---

## Development Setup

```bash
git clone https://github.com/Swayam26-rwt/SecureID.git
cd SecureID
pip install -e ".[dev]"          # editable install + pytest, mypy, ruff
python -m pytest tests/ -v       # verify 61 tests pass before making changes
```

---

## Non-Negotiable Design Constraints

These constraints exist to keep SecureID deployable in air-gapped, edge, and embedded environments.

- **Zero External Dependencies**: The core Python package (`biometric_engine/`) must run using only standard library primitives (`math`, `typing`, `hashlib`, `collections`, `datetime`, `dataclasses`). Do **not** introduce NumPy, SciPy, OpenCV, or any third-party library.
- **Self-Contained Frontend**: `index.html` must remain fully functional via direct `file://` execution as well as any static HTTP server, without requiring Node.js, npm, webpack, or external CDNs. All ML logic in `app.js` must be a self-contained port.
- **Strict Typing**: All new Python functions must include complete type hints (`from __future__ import annotations`). Run `python -m mypy biometric_engine/ --strict` before submitting a PR.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`.
- **ISO/IEC 30107-3 Terminology**: New code must use standard terminology — BRS, PAD, CDS, FMR, FNMR, TMR, BRT, BAM — not informal equivalents.

---

## PR Checklist

Before opening a pull request, verify:

- [ ] `python -m pytest tests/ -v` — all 61 tests pass
- [ ] `python -m mypy biometric_engine/ --strict` — no type errors
- [ ] `python -m ruff check biometric_engine/` — no lint warnings
- [ ] No new external dependencies introduced
- [ ] New functions have full docstrings with Args/Returns/Reference
- [ ] Commit messages follow Conventional Commits
- [ ] CHANGELOG.md updated under `## [Unreleased]`

---

## Code Style

- Line length: 100 characters (configured in `pyproject.toml`)
- Use `# ── Section ───────` dividers for logical grouping within modules
- Keep public APIs minimal; internals should be `_prefixed`
- Prefer dataclasses with `frozen=True` for immutable result types

---

## Adding a New ML Algorithm

1. Implement it in `biometric_engine/` using only the standard library
2. Add a corresponding port in `assets/app.js` with an identical mathematical signature
3. Expose it in the `FusionResult` or `RecognitionResult` dataclass if it produces a new score field
4. Reference the academic paper or standard in the module docstring
5. Add at least 3 unit tests covering edge cases (empty input, single sample, degenerate distributions)

---

## Reporting Issues

Use [GitHub Issues](https://github.com/Swayam26-rwt/SecureID/issues). Include:
- Python version (`python --version`)
- Exact error traceback
- Minimal reproduction steps
