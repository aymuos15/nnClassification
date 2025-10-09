# Getting Started Overview

This section provides lightweight companions to the unified workflow. Follow the workflow for the full narrative; dip into these pages for quick reminders and platform-specific tips.

---

## Quick Navigation

| Guide | Purpose | When to read |
| --- | --- | --- |
| [Installation](installation.md) | Hardware/software requirements, concise install recipe, platform notes, troubleshooting highlights | Setting up a new machine or debugging installs |
| [Data Preparation](data-preparation.md) | Required `raw/` + `splits/` layout, `ml-split` customization, validation utilities | Organising any dataset before training |
| [Quick Start](quick-start.md) | 5-minute sample run, sanity checks, pointers back to the workflow | First-look experience or smoke test |

For the sequential end-to-end workflow (install → prep → tune → train → infer → export), start with **[Workflow Guide](../workflow.md)**.

---

## Suggested First Steps

1. **Install** the package (Workflow Step 1) and confirm `ml-train --help` works.
2. **Organise** your dataset using the layout from the Data Preparation guide (Workflow Step 2).
3. **Run** the Quick Start commands to train the bundled ants vs bees example.

After that, continue through the workflow for cross-validation, tuning, inference, and export.
