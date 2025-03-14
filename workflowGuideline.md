# Project Workflow Guideline

This document describes the branching strategy, commit message conventions, and clean history maintenance for this project to ensure consistency and clarity across all contributions.

---
## Branching Strategy

The project follows a **branch-based workflow** to organize features, fixes, and releases efficiently. Below is the branching strategy to be followed:
### Branch Types

- **`main`**  
  - Contains the latest stable release code.
  - Only merged after review and passing tests.
  - No direct commits allowed.

- **`feature/{feature-name}`**  
  - Used for working on individual features.
  - Branch naming convention: `feature/login-ui` or `feature/integration-x-api`.

- **`bugfix/{issue-number}-{description}`**  
  - Dedicated for bug fixes.
  - Example: `bugfix/42-login-issue`.

- **`hotfix/{issue-number}-{description}`**  
  - Used for critical, time-sensitive patches to the `main` branch.
  - Example: `hotfix/99-crash-on-launch`.
---

## Commit Message Conventions

### Format

```
<type>(<scope>): <subject>

<body>  # Optional but recommended
<footers>  # Optional: references issues or breaking changes
```

#### **Examples:**
- `feat(auth): add login page UI`
- `fix(api): correct endpoint URL for orders`
- `docs(readme): update installation instructions`

### **Commit Types:**
- **feat:** A new feature
- **fix:** A bug fix
- **docs:** Documentation changes only
- **style:** Code style improvements (e.g., formatting)
- **refactor:** Code changes that neither fix a bug nor add a feature
- **test:** Adding or improving tests
- **chore:** Routine tasks like dependency updates

### **Scope:**  
Optional but recommended. Indicates the area of the codebase affected (e.g., `auth`, `ui`, `api`).

### **Subject:**  
A short summary (imperative form, present tense) describing the change.

---

## Pull Requests and Merging Process

1. **Open a Pull Request (PR)**  
   - Target: `main` for  feature/bugfix/hotfixes branches

2. **Review Process:**  
   - At least one team member must review your PR.

4. **Squash and Merge:**  
   Use **squash and merge** to ensure a linear history.

---

## Best Practices

- **Commit frequently:** Make small, incremental commits with meaningful messages.
- **Keep PRs focused:** Avoid working on multiple issues in a single branch.
- **Update branches regularly:** Rebase frequently to avoid large merge conflicts.
