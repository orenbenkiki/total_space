'''
Run test cases and compare actual results files with expected result files.
'''

from typing import *

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


import glob
import filecmp
import pytest
import simple_model
import total_space


def pytest_assertrepr_compare(op: str, left: Any, right: Any) -> List[str]:
    '''
    Report a nice error when the assertion fails.
    '''


@pytest.mark.parametrize('case', [name[6:-5] for name in glob.glob('tests/*.case')])
def test_case(case: str) -> None:
    '''
    Compare actual vs. expected file for a specific test case.
    '''
    flags = open('tests/%s.case' % case).read().split()
    sys.argv = ['test', '-o', 'tests/%s.actual' % case] + flags
    total_space.main(flags=simple_model.flags, model=simple_model.model)
    result = filecmp.cmp('tests/%s.expected' % case, 'tests/%s.actual' % case)
    if result:
        assert result  # Count successful test assertions
    else:
        pytest.fail('The file: tests/%s.actual is different from the file: tests/%s.expected' % (case, case), False)
