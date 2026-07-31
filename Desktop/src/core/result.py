# src/core/result.py

from dataclasses import dataclass

@dataclass
class TaskResult:

    success: bool

    task_name: str

    message: str

    output_file: str | None = None

    accuracy: float | None = None

    stdout: str | None = None

    stderr: str | None = None