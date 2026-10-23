# Dataset Directory Structure

This folder contains the clinical and in-the-wild datasets used in this research.
Due to dataset licenses and size constraints, raw and processed medical imagery are not tracked in Git.

## Expected Directory Layout:

```text
data/
├── CaDIS/
│   └── ...
├── Cataract-1K/
│   └── ...
├── Cataract-LMM/
│   └── ...
├── YouTube_OOD/
│   └── ...
└── processed/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## Dataset Sources:
- **CaDIS**: [Grand Challenge CaDIS](https://cataracts.grand-challenge.org/)
- **Cataract-1K**: [Grand Challenge Cataract-1K](https://cataract1k.grand-challenge.org/)
- **Cataract-LMM**: Multi-center clinical dataset
- **YouTube Cohort**: Unconstrained out-of-distribution evaluation split
