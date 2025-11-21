# Appendix Usage Instructions

This document explains how to integrate the implementation details appendix into your LaTeX paper.

## Files Created

1. **`paper_appendix_implementation.tex`** - Full detailed appendix (recommended for journals with no space constraints)
2. **`paper_appendix_short.tex`** - Shorter version (for space-constrained venues)
3. **`additional_references.bib`** - BibTeX entries for all cited libraries and software

## Integration Options

### Option 1: Full Appendix (Recommended for Arxiv/Technical Reports)

Add to your main document before `\end{document}`:

```latex
% ... existing content ...

\appendix

\section{PABBO Hyperparameters}
% Your existing PABBO appendix content

\input{paper_appendix_implementation}

\bibliographystyle{plain}
\bibliography{references,additional_references}

\end{document}
```

### Option 2: Short Appendix (For Space-Constrained Journals)

Add to your main document:

```latex
% ... existing content ...

\appendix

\section{PABBO Hyperparameters}
% Your existing PABBO appendix content

\input{paper_appendix_short}

\bibliographystyle{plain}
\bibliography{references,additional_references}

\end{document}
```

### Option 3: Separate Supplementary Material

Keep the appendix as a separate document:

```latex
% supplementary.tex
\documentclass{article}
\usepackage{hyperref}
\usepackage{amsmath}

\title{Supplementary Material: Topic Modelling Black Box Optimization}
\author{[Authors]}

\begin{document}
\maketitle

\input{paper_appendix_implementation}

\bibliographystyle{plain}
\bibliography{references,additional_references}

\end{document}
```

## Merging Bibliography Files

If you have an existing `references.bib` file, you can either:

1. **Merge the files manually:**
   ```bash
   cat additional_references.bib >> references.bib
   ```

2. **Use multiple bibliography files:**
   ```latex
   \bibliography{references,additional_references}
   ```

## Customization

### Adjusting Detail Level

The implementation appendix is modular. You can comment out sections you don't need:

```latex
% \subsection{Parallel Execution}  % Comment out if not relevant
```

### Adding Missing Information

Key sections you may want to customize:

1. **Line 288** in `paper_appendix_implementation.tex`: Update the GitHub repository URL when available
2. **Line 404**: Update the SABBO implementation status if you complete it
3. **Dataset descriptions**: Add specific details about your corpora (lines 186-202)

## Important Note on SABBO

The appendix explicitly states that **SABBO results are based on simulated data**, not actual implementation. This is mentioned in:
- `paper_appendix_implementation.tex` (Section "SABBO")
- `paper_appendix_short.tex` (paragraph "SABBO")

If you complete the SABBO implementation, update these sections accordingly.

## Compilation

Standard LaTeX compilation:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk:

```bash
latexmk -pdf main.tex
```

## Checklist Before Submission

- [ ] Verify all library versions match your actual environment
- [ ] Update GitHub repository URL (or remove if not public)
- [ ] Confirm dataset descriptions match your actual data
- [ ] Update SABBO implementation status
- [ ] Check that all cross-references work (especially Figure references)
- [ ] Verify bibliography compiles without errors
- [ ] Confirm file paths in code snippets match your structure

## Questions?

If you need to adjust any section or have questions about integration, refer to the comments in the LaTeX files or check the original implementation files in:
- `lda_hyperopt/`
- `PABBO/`
- `lda_parallel.py`
