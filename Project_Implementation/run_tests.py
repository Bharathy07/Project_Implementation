"""Simple test runner to execute test functions in tests/test_patientid.py
This avoids depending on pytest being installed in the environment.
"""
import sys
from pathlib import Path
import importlib.util
import traceback

REPO_ROOT = Path(__file__).resolve().parents[0]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

# discover all test_*.py files in tests/ and run their test_ functions
import glob
results = {"passed": [], "failed": []}
for test_file in glob.glob(str(REPO_ROOT / "tests" / "test_*.py")):
    spec = importlib.util.spec_from_file_location(test_file, test_file)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR importing {test_file}: {e}")
        traceback.print_exc()
        results["failed"].append((test_file, str(e)))
        continue
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            if callable(fn):
                try:
                    fn()
                    print(f"PASSED: {name} ({test_file})")
                    results["passed"].append((test_file, name))
                except AssertionError as e:
                    print(f"FAILED: {name} ({test_file}) - AssertionError: {e}")
                    traceback.print_exc()
                    results["failed"].append((test_file + ":" + name, str(e)))
                except Exception as e:
                    print(f"ERROR: {name} ({test_file}) - Exception: {e}")
                    traceback.print_exc()
                    results["failed"].append((test_file + ":" + name, str(e)))

print("\nSUMMARY:")
print(f"Passed: {len(results['passed'])}")
print(f"Failed: {len(results['failed'])}")
if results['failed']:
    for name, msg in results['failed']:
        print(f" - {name}: {msg}")

# exit with non-zero code on failure for CI friendliness
if results['failed']:
    raise SystemExit(1)
else:
    raise SystemExit(0)
