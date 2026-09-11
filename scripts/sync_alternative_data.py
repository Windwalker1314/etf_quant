import sys

from steadyquant.alternative_data import pilot, sync

if "--pilot" in sys.argv:
    pilot()
else:
    sync()
