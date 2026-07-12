# Commit Message Guidelines

To ensure the reproducibility of our research and maintain a clean, scannable project history, all contributors must follow these structured commit rules. This helps us quickly track when mathematical logic changed versus when engineering optimizations were introduced.

---

## 1. The Structure of a Commit

We follow a structured format based on Conventional Commits, extended for machine learning research:

```text
<type>(<scope>): <short description in present tense>

[optional body: explain the WHY, mathematical changes, or hyperparameter details]

[optional footer: reference issue numbers, paper sections, or PRs]
```

### Example:
```text
math(priors): implement historically consistent physics loss invariant

Added the Riemann manifold constraint to the loss function to ensure historical 
consistency across long-term chaotic horizons. Matches Equation 4.2 in the paper draft.

Fixes #42
```

---

## 2. Allowed Commit Types (`<type>`)

Choose the type that best describes the primary intent of your change:

### 🔬 Research & Modeling Types
*   **`math`**: Changes to the core underlying equations, loss functions, network constraints, or theorems.
*   **`arch`**: Changes to the neural network architecture (e.g., adding layers, changing activation functions, swapping attention blocks).
*   **`data`**: Updates to data pipelines, synthetic dynamical system generators, normalization strategies, or feature engineering.
*   **`hparam`**: Tweaks to baseline hyperparameter files, training seeds, or training run configurations.

### 🛠️ Software Engineering Types
*   **`feat`**: A new software feature for the package (e.g., adding a new evaluator, checkpoint saver, or CLI argument).
*   **`fix`**: A bug fix (e.g., resolving a shape mismatch, fixing a gradient explosion bug, or correcting a device allocation issue).
*   **`docs`**: Documentation only updates (e.g., updating the README, adding mathematical explanations to docstrings, writing API guides).
*   **`perf`**: Code changes that improve runtime performance (e.g., vectorizing a loop, optimizing CUDA memory, multi-GPU support).
*   **`refactor`**: Code changes that neither fix a bug nor add a feature, but improve code readability or structure.
*   **`test`**: Adding missing tests or correcting existing validation scripts.

---

## 3. Scopes (`<scope>`)

Scopes define *where* the change happened in the codebase. Common scopes for our package include:
*   `lorenz` / `navier-stokes` / `pendulum` (specific dynamical systems)
*   `priors` (historical consistency / physical constraints)
*   `forecaster` (the main rollout/prediction loop)
*   `trainer` (optimization loop, learning rate schedulers)
*   `utils` (saving, loading, visualization scripts)

*If a commit affects multiple scopes, you can omit the parenthesis or use a general scope.*

---

## 4. The Golden Rules for Writing Commit Messages

1.  **Use the Imperative Mood:** Write the short description as if you are commanding the codebase to do something.
    *   ❌ `fixed shape mismatch in jacobian`
    *   ❌ `fixes shape mismatch in jacobian`
    *   ✅ `fix shape mismatch in jacobian`
2.  **No Trailing Periods:** Do not end the first summary line with a period (`.`).
3.  **Keep it Short:** The first line must be under **50 characters** if possible, and absolutely never exceed **72 characters**.
4.  **Explain the "Why" in the Body:** If you change a constant, a learning rate, or an architectural block, use the commit body to explain *why* that change was necessary (e.g., "Prevents chaotic divergence after 500 timesteps").
5.  **Separate with Blank Lines:** Always leave a blank line between the summary line, the body, and the footer.

---

## 5. Summary Cheat-Sheet

| Type | Intent | Example |
| :--- | :--- | :--- |
| **math** | Equations / Loss definitions | `math(loss): introduce time-reversibility penalty` |
| **arch** | Model layers / Embeddings | `arch(node): swap MLP with Graph Conv in Neural ODE` |
| **data** | Dataloaders / Noise generation | `data(lorenz): add high-frequency Gaussian noise` |
| **fix** | Software or execution bugs | `fix(trainer): cast target states to float32 on CUDA` |
| **feat** | New user-facing tool/feature | `feat(viz): add phase space attractor plotting utility` |

Thank you for keeping our research codebase clean, reproducible, and structured!

Rockefeller, PhD
