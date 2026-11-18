"""
Spustí všechny pipeline postupně. Chyba v jednom skriptu nezastaví ostatní.

Každý skript běží jako samostatný proces se stejným Pythonem (stejné prostředí),
jeho výstup jde do konzole i do logu. Na konci se vypíše souhrn.

Spuštění (ze složky se skripty):
    python run_all.py
    python run_all.py cph_pipeline.py mainz_pipeline.py   # jen vybrané
"""
import os
import subprocess
import sys
import time
from datetime import datetime
import faulthandler, sys

SCRIPTS = [
    # "prague_pipeline.py",
    # "cph_pipeline.py",
    "mainz_pipeline.py",
    "amstr_pipeline.py",
    "bochum_pipeline.py",
    "genova_pipeline.py",
]

# Složka, ze které se skripty spouští. Musí v ní být podsložka "executables"
# (MorphoBox se volá relativní cestou executables\shs_dl_morphometry.exe).
WORKDIR = os.path.dirname(os.path.abspath(__file__))


def run(script: str, log_dir: str) -> tuple[int, float]:
    log_path = os.path.join(log_dir, script.replace(".py", ".log"))
    start = time.time()
    print(f"\n{'=' * 70}\n▶ {script}  ({datetime.now():%H:%M:%S})\n  log: {log_path}\n{'=' * 70}", flush=True)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(WORKDIR, script)],
                cwd=WORKDIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:  # průběžně do konzole i do logu
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    except Exception as err:  # nepodařilo se ani spustit
        print(f"!! {script}: nelze spustit: {err}", flush=True)
        code = -1
    return code, time.time() - start


def main():
    faulthandler.enable(file=sys.stderr, all_threads=True)
    scripts = sys.argv[1:] or SCRIPTS
    log_dir = os.path.join(WORKDIR, "run_all_logs", datetime.now().strftime("%Y_%m_%d_%H%M"))
    os.makedirs(log_dir, exist_ok=True)

    results = []
    for script in scripts:
        if not os.path.isfile(os.path.join(WORKDIR, script)):
            print(f"!! {script} neexistuje, přeskakuji", flush=True)
            results.append((script, None, 0))
            continue
        code, secs = run(script, log_dir)
        results.append((script, code, secs))

    print(f"\n{'=' * 70}\nSOUHRN\n{'=' * 70}")
    for script, code, secs in results:
        status = "nenalezen" if code is None else ("OK" if code == 0 else f"CHYBA (kód {code})")
        print(f"  {script:25s} {status:18s} {secs / 60:6.1f} min")
    print(f"\nLogy: {log_dir}")
    print("Pozn.: chyby jednotlivých subjektů jsou v logu každého centra (derivatives/<datum>/logs).")


if __name__ == "__main__":
    main()
