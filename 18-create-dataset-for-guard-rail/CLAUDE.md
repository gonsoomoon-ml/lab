# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Validate dataset integrity (duplicates, fields, label distribution)
python scripts/validate_dataset.py
```

## Key Context

- **Language**: All data is in Korean (한국어)
- **Policy**: Musinsa strict return policy - only defects/damage/wrong items allowed
- **Labeling**: Unsafe = block fraudulent intent, Safe = pass honest questions to AI agent

See README.md for full documentation.
