"""Unified Contributor Validation Suite.

Runs all offline verification checks (artifact integrity, backend routing,
GPU subprocess isolation, and stratified capping regression tests) in sequence
without requiring an NVIDIA GPU or making live API calls.

Usage:
    python validate_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

CHECKS = [
    ("validate_benchmark_package.py", "Modular benchmark.* package imports & subsystem contracts"),
    ("validate_v3_artifact.py", "Artifact integrity, source hashes, and published table checks"),
    ("validate_v3_backends.py", "Backend estimator routing matrix & constructor double checks"),
    ("validate_gpu_process.py", "Multi-GPU subprocess device masking & process reaping checks"),
    ("validate_sampling.py", "Proportional capping & stratified split regression checks"),
]

def main():
    root = Path(__file__).resolve().parent
    print("=" * 72)
    print("JEV-VS-ML — Offline Contributor Validation Suite (Protocol 3.0.1)")
    print("=" * 72)
    
    passed, failed = 0, []
    total_start = time.perf_counter()
    
    for script_name, description in CHECKS:
        script_path = root / script_name
        if not script_path.exists():
            print(f"[-] MISSING: {script_name}")
            failed.append((script_name, "File not found"))
            continue
            
        print(f"\n[RUNNING] {script_name} — {description}...")
        start = time.perf_counter()
        
        proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        
        if proc.returncode == 0:
            print(f"[PASS] {script_name} ({elapsed:.2f}s)")
            if proc.stdout.strip():
                for line in proc.stdout.strip().splitlines():
                    print(f"       {line}")
            passed += 1
        else:
            print(f"[FAIL] {script_name} ({elapsed:.2f}s) — Exit code {proc.returncode}")
            if proc.stderr.strip():
                print("--- STDERR ---")
                print(proc.stderr.strip())
            if proc.stdout.strip():
                print("--- STDOUT ---")
                print(proc.stdout.strip())
            failed.append((script_name, proc.stderr.strip() or f"Exit code {proc.returncode}"))
            
    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 72)
    print(f"Summary: {passed}/{len(CHECKS)} checks passed in {total_elapsed:.2f}s")
    if failed:
        print("\nFailed checks:")
        for name, err in failed:
            print(f"  - {name}: {err[:100]}")
        print("=" * 72)
        sys.exit(1)
    else:
        print("All offline checks PASSED. Ready for pull request review.")
        print("=" * 72)
        sys.exit(0)

if __name__ == '__main__':
    main()
