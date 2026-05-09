"""Entry point for the CSC 7644 quantum bug classification project."""


def main():
    """Print project usage instructions."""
    print("Fine-Tuning LLMs for Bug Detection in Quantum Software")
    print("Main notebook: notebooks/quantum-vs-classical-bug-detection-Final.ipynb")
    print("Install dependencies with: pip install -r requirements.txt")
    print()
    print("Common commands:")
    print("  python scripts/run_training.py        # full training run (GPU recommended)")
    print("  python scripts/run_evaluation.py      # print saved metrics")
    print("  python scripts/generate_figures.py    # regenerate PNG figures in figures/")


if __name__ == "__main__":
    main()
