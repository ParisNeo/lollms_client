---
name: git_workflow_mastery
title: Git Version Control and Safe Branching
category: version_control
tags: [git, branching, commits, stash, safety, workflow]
required_tools: [tool_git_status, tool_git_diff, tool_git_commit, tool_git_checkout]
description: Safe, professional Git workflow guidelines, branch isolation, and uncommitted change protection.
visibility: loadable
---

# Git Workflow Mastery Skill

## Safe Git Operations
1. **Check Status First**: Always run `git status` before making branching decisions.
2. **Branch Isolation**: Never perform major refactoring or destructive edits directly on `main` or `master`. Create an isolated feature branch:
   `git checkout -b feature/<task-name>`
3. **Protecting Dirty Working Trees**: If uncommitted changes exist, NEVER switch branches without stashing (`git stash`) or committing them first.
4. **Atomic Commits**: Write clear, descriptive commit messages summarizing the "why" and "what":
   `feat(auth): implement JWT token rotation and validation`