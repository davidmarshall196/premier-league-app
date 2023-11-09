#import pytest
#%run -m pytest  -W ignore::DeprecationWarning -W ignore::FutureWarning

import runpy
# Execute the run_tests.py script
r = runpy.run_path('run_tests.py')



