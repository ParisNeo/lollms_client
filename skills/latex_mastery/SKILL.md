---
title: "LaTeX Document Writing, Scientific Typesetting, and Autonomous Compilation Mastery"
description: "Comprehensive guide to writing publication-grade LaTeX (.tex) documents, equations, TikZ diagrams, bibliographies, and autonomous compilation workflows."
category: "typesetting_engineering"
tags: [latex, pdflatex, biblatex, tikz, academic_writing, typesetting, pdf]
visibility: loadable
modifiable: true
---

# Professional LaTeX Engineering & Scientific Typesetting

This skill teaches how to formulate clean, modular, publication-quality LaTeX documents (`.tex`), mathematical equations, IEEE/ACM format papers, TikZ architecture diagrams, and automate zero-error compilation with Python.

---

## 🎨 Part 1: High-Level Typographic Standards & Package Arsenal

### Essential Package Arsenal (Standard Preamble)
```latex
\documentclass[11pt,a4paper]{article}

% --- Core Geometry & Fonts ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{microtype} % Superior kerning and character protrusion

% --- Mathematics & Science ---
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{mathtools}
\usepackage{siunitx} % Proper unit formatting: \SI{100}{\kilo\meter\per\hour}

% --- Tables & Arrays ---
\usepackage{booktabs} % Publication tables: \toprule, \midrule, \bottomrule
\usepackage{tabularx} % Dynamic auto-width columns (X)
\usepackage{multirow}

% --- Figures & Graphics ---
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,calc}

% --- Code Listings & Algorithms ---
\usepackage{listings}
\usepackage{xcolor}

% --- Hyperlinks & References ---
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=green!50!black,
    urlcolor=blue!80!black
}
```

### Table Best Practices (Booktabs Rule)
- ❌ **Never** use vertical rules (`|`).
- ❌ **Never** use double horizontal rules.
- ✅ **Always** use `\toprule`, `\midrule`, and `\bottomrule`.
- ✅ Right-align numeric columns; left-align text descriptions.

---

## 💻 Part 2: Complete Publication-Grade Article Template

```latex
\documentclass[11pt,a4paper]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{tikz}
\usetikzlibrary{positioning,shapes.geometric,arrows.meta}
\usepackage{hyperref}

\title{\textbf{Autonomous Agent Orchestration in Heterogeneous Workspaces: A Two-Tier Architectural Evaluation}}
\author{
    \textbf{ParisNeo} \quad \textbf{Lollms Research Team}\\
    \small Sovereign AI Systems Laboratory\\
    \small \texttt{contact@lollms.com}
}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
We introduce a decoupled, two-tier execution doctrine separating cognitive planning from atomic tool dispatch in large language model (LLM) agents. By enforcing sandbox confinement, dynamic context budgeting, and formal state verification, our architecture achieves robust verification loops while maintaining zero-amnesia state transitions. Experimental results demonstrate a 34\% reduction in token consumption and complete elimination of phantom tool invocation loops.
\end{abstract}

\section{Introduction}
Modern autonomous agents frequently suffer from context pollution when long-running tool execution outputs overwhelm working memory budgets \cite{vaswani2017attention}. In this paper, we formalize the \textit{Decoupled Execution Protocol} (DEP).

\section{Mathematical Formulation}
Let $\mathcal{S}$ denote the state space of the discussion workspace, and let $\mathcal{A}$ represent the bounded set of executable tool operations. The state transition probability given observation $o_t$ is defined as:

\begin{equation}
P(s_{t+1} \mid s_t, a_t) = \sigma\left( \frac{\mathbf{W}_s s_t + \mathbf{W}_a a_t}{\sqrt{d_k}} \right)
\end{equation}

where $\mathbf{W}_s$ and $\mathbf{W}_a$ denote the state and action projection matrices respectively, and $\sigma(\cdot)$ is the softmax activation function.

\section{System Architecture}
Figure~\ref{fig:architecture} illustrates the high-level orchestration workflow between the Orchestrator tier and the Worker sandboxes.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.8cm,
    block/.style={rectangle, draw=blue!80, fill=blue!10, thick, rounded corners, minimum width=3.5cm, minimum height=1.0cm, align=center},
    worker/.style={rectangle, draw=green!80!black, fill=green!10, thick, rounded corners, minimum width=3.5cm, minimum height=1.0cm, align=center},
    arrow/.style={-Latex, thick}
]
    \node[block] (orch) {\textbf{Orchestrator Tier}\\(Persistent Context)};
    \node[worker, below=of orch] (worker) {\textbf{Worker Sandbox}\\(Disposable Context)};
    \node[block, right=2.5cm of worker] (tools) {\textbf{Tool Engine}\\(Filesystem \& Shell)};

    \draw[arrow] (orch) -- node[right]{\small Task + Files} (worker);
    \draw[arrow] (worker) -- node[above]{\small Tool Call} (tools);
    \draw[arrow] (tools) |- node[below right]{\small Result} (worker);
    \draw[arrow] (worker.west) to[out=180,in=180] node[left]{\small Plain Report} (orch.west);
\end{tikzpicture}
\caption{Decoupled Two-Tier Execution Flow Diagram}
\label{fig:architecture}
\end{figure}

\section{Empirical Evaluation}
Table~\ref{tab:benchmarks} summarizes the benchmark performance across 100 complex multi-step coding missions.

\begin{table}[htbp]
\centering
\small
\caption{Performance Comparison Across Agent Execution Architectures}
\label{tab:benchmarks}
\begin{tabularx}{\textwidth}{lcccc}
\toprule
\textbf{Architecture Model} & \textbf{Success Rate (\%)} & \textbf{Avg. Rounds} & \textbf{Context Fill (\%)} & \textbf{Loop Interceptions} \\
\midrule
Single-Tier Monolithic      & 71.4                       & 14.2                 & 86.2                       & 19 \\
Standard ReAct Loop         & 78.9                       & 11.5                 & 74.0                       & 11 \\
\textbf{Two-Tier DEP (Ours)} & \textbf{96.2}              & \textbf{6.8}         & \textbf{32.4}              & \textbf{0} \\
\bottomrule
\end{tabularx}
\end{table}

\section{Conclusion}
The Two-Tier Decoupled Execution Protocol guarantees robust convergence in autonomous programming environments without context bloat or phantom loops.

\begin{thebibliography}{9}
\bibitem{vaswani2017attention}
A.~Vaswani et~al., ``Attention is all you need,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2017, pp. 5998--6008.
\end{thebibliography}

\end{document}
```

---

## ⚙️ Part 3: Autonomous Compilation Script in Python

This Python helper compiles a `.tex` file to `.pdf` using `pdflatex` or `latexmk`, executing multiple passes to resolve citations and cross-references cleanly.

```python
import subprocess
import shutil
from pathlib import Path

def compile_latex(tex_file: str, output_dir: str = None) -> dict:
    tex_path = Path(tex_file).resolve()
    if not tex_path.exists():
        return {"success": False, "error": f"File '{tex_file}' not found."}

    work_dir = Path(output_dir).resolve() if output_dir else tex_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # Detect compilation engine
    compiler = "latexmk" if shutil.which("latexmk") else "pdflatex"
    
    if not shutil.which(compiler):
        return {
            "success": False,
            "error": "Neither 'latexmk' nor 'pdflatex' was found on your system path. Please install TeX Live, MiKTeX, or MacTeX."
        }

    try:
        if compiler == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", f"-output-directory={work_dir}", str(tex_path)]
            passes = 1
        else:
            cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={work_dir}", str(tex_path)]
            passes = 2  # Run twice for cross-references/TOC

        for p in range(passes):
            res = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, check=False)
            if res.returncode != 0 and p == passes - 1:
                # Extract clean error lines from LaTeX output
                log_lines = res.stdout.splitlines()
                errors = [l for l in log_lines if l.startswith("!") or "Error" in l]
                return {
                    "success": False,
                    "error": f"Compilation failed:\n" + "\n".join(errors[:10]),
                    "raw_log": res.stdout[-2000:]
                }

        pdf_path = work_dir / f"{tex_path.stem}.pdf"
        return {
            "success": True,
            "pdf_path": str(pdf_path),
            "output": f"Successfully compiled {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)."
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = compile_latex("paper.tex")
    print(result)
```