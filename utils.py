from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
