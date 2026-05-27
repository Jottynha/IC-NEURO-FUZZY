from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "src/preprocessing.py",
    "src/mlp_classifier.py",
    "src/rbm_logistic_classifier.py",
    "src/mamdani_fuzzy_classifier.py",
    "src/anfis_classifier.py",
]

def run_script(path: str) -> int:
    print(f"\n>>> Executando: {path}")
    start = time.time()
    p = subprocess.run([sys.executable, path], cwd=ROOT)
    elapsed = time.time() - start
    print(f"<<< Concluído: {path} (tempo {elapsed:.1f}s)")
    return p.returncode

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        rc = run_script(script)
        if rc != 0:
            print(f"Script {script} retornou código {rc}. Interrompendo.")
            break

if __name__ == "__main__":
    main()
