# Paper Appendix Files

This directory contains all files related to the Implementation Details appendix for the paper "Topic Modelling Black Box Optimization".

## 📁 Files Overview

### LaTeX Files (for paper)
- **`paper_appendix_implementation.tex`** (21 KB) - Full detailed appendix
- **`paper_appendix_short.tex`** (4 KB) - Condensed version for space-constrained journals
- **`additional_references.bib`** (5 KB) - Bibliography entries for all cited software

### Documentation (for authors)
- **`APPENDIX_README_RU.md`** (8.6 KB) - Detailed guide in Russian
- **`APPENDIX_USAGE_INSTRUCTIONS.md`** (3.6 KB) - Integration instructions in English
- **`APPENDIX_SUMMARY.md`** (6.1 KB) - Quick summary of all content

## 🚀 Quick Start

### To add full appendix to your paper:

```latex
\appendix
\input{paper_appendix/paper_appendix_implementation}
\bibliography{references,paper_appendix/additional_references}
```

### To add short version:

```latex
\appendix
\input{paper_appendix/paper_appendix_short}
\bibliography{references,paper_appendix/additional_references}
```

## 📖 What to Read First

1. **Start here:** `APPENDIX_SUMMARY.md` - Overview of everything
2. **For details:** `APPENDIX_README_RU.md` - Full guide in Russian
3. **For integration:** `APPENDIX_USAGE_INSTRUCTIONS.md` - How to add to paper

## ⚠️ Before Submission

Check these items:
- [ ] Update GitHub URL in appendix (line 288 of full version)
- [ ] Verify SABBO status (simulated vs implemented)
- [ ] Confirm library versions match your environment
- [ ] Review dataset descriptions

## 📊 Content Covered

- ✅ Software environment (Python, libraries)
- ✅ GA implementation (DEAP)
- ✅ ES implementation (DEAP)
- ✅ PABBO implementation (PyTorch + official code)
- ✅ SABBO status (simulated - explicitly stated)
- ✅ Data preprocessing
- ✅ Logging & results
- ✅ Parallel execution
- ✅ Reproducibility checklist

## 🔗 File Dependencies

```
Main LaTeX document
├── \input{paper_appendix/paper_appendix_implementation.tex}
│   └── Uses references from:
│       └── additional_references.bib
└── Bibliography compiles all together
```

---

**Created:** November 21, 2024
**Location:** `/Users/draiqws/Llabs/paper_appendix/`
**Purpose:** Keep appendix files organized and separate from main project files
