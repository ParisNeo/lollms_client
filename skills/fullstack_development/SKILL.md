---
name: fullstack_development
title: Full-Stack Web and Application Engineering
category: software_engineering
tags: [fullstack, frontend, backend, api, database, architecture, testing]
description: Architecture, conventions, and implementation workflows for full-stack systems, backend APIs, modern frontends, and database migrations.
---

# Full-Stack Development Skill

## Architecture Patterns
1. **Decoupled Layers**: Enforce clean separation of concerns:
   - Data Layer: ORM models, migrations, repository patterns.
   - Service / Domain Layer: Business logic, validation, core algorithms.
   - API / Transport Layer: FastAPI, Flask, Express, or Next.js endpoints.
   - Client / UI Layer: Reactive components, state management, semantic HTML, responsive CSS.
2. **Type Safety & Contracts**:
   - Python: Pydantic schemas, strict type hints, dataclasses.
   - TypeScript/JS: Strict interfaces, Zod validation.
3. **Testing & Validation**:
   - Write automated unit tests for business logic before considering tasks done.
   - Run test suites (`pytest`, `npm test`) using `tool_execute_shell_command` to verify functionality.