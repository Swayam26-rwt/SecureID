# Contributing to SecureID — Offline Facial Recognition Engine

Thank you for your interest in contributing! This document outlines guidelines for submitting bug fixes, performance improvements, and algorithmic enhancements.

## Development Workflow

1. **Prerequisites**: Python 3.9+ and a modern web browser for frontend verification.
2. **Setup**:
   ```bash
   git clone https://github.com/Swayam26-rwt/SecureID.git
   cd SecureID
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. **Running Tests**:
   ```bash
   python3 -m unittest discover -s tests
   ```

## Design Principles

- **Zero External Dependencies**: The core Python package must run using only standard library primitives (`math`, `typing`, `collections`, etc.).
- **Self-Contained Frontend**: The browser demo in `index.html` must remain fully functional via direct `file://` execution as well as static HTTP servers, without requiring Node.js build pipelines or external bundlers.
- **Strict Typing**: All new Python functions must feature complete type hints (`from __future__ import annotations`).
- **Commit Messages**: Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
