import json
import subprocess
import sys

TASKS = [
    "bace_c",
    "bbbp",
    "clintox",
    "hiv",
    "tox21",
    "sider",
    "esol",
    "bace_r",
    "lipo",
    "freesolv",
    "clearance",
]


def main():
    summary = {}
    for task in TASKS:
        cmd = [sys.executable, "-m", "chemberta.training.train", f"task={task}"]
        print("Running:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, check=False)
        summary[task] = {"returncode": proc.returncode}

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
