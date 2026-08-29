import sys
print(sys.version)
try:
    import pymoo
    print("pymoo", getattr(pymoo, '__version__', 'unknown'))
except Exception as e:
    print('pymoo import error:', e)
