# Topic Modelling Black Box Optimization

Comparison of black-box optimization algorithms (GA, ES, PABBO, SABBO) for LDA hyperparameter tuning.

## Quick Start

```bash
pip install -r requirements.txt
cd lda_hyperopt
python run.py --data ../data/X_20news_val_bow.npz --iterations 20
```

## Project Structure

```
Llabs/
├── lda_hyperopt/          # Main optimization code
├── PABBO/                 # PABBO implementation
├── data/                  # Text corpora
├── paper_appendix/        # Paper appendix (LaTeX)
└── docs/                  # Documentation
```

## Documentation

- **Paper appendix:** `paper_appendix/README.md`
- **All documentation:** `docs/README.md`
- **Detailed project info:** `docs/PROJECT_README.md`

## Algorithms

| Algorithm | Implementation |
|-----------|----------------|
| GA | `lda_hyperopt/optimizers/ga.py` |
| ES | `lda_hyperopt/optimizers/es.py` |
| PABBO | `lda_hyperopt/optimizers/pabbo_full.py` |
| SABBO | *Simulated (not implemented)* |

---

For detailed information, see `docs/` folder.
