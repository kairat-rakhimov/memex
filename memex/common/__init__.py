"""Cross-cutting infrastructure shared across the app.

- ``config`` — env-driven Settings.
- ``logging_setup`` — root logger configuration.
- ``schemas`` — /run request/response + trace-event models.
- ``trace`` — per-request JSONL agent-trace writer.
"""
