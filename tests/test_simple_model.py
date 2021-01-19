'''
Run test cases and compare actual results files with expected result files.
'''

# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from typing import *

import filecmp
import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# pylint: disable=wrong-import-position
import total_space
import total_space.simple_model as simple_model


@pytest.mark.parametrize('case', [name[6:-5] for name in glob.glob('tests/*.case')])
def test_case(case: str) -> None:
    '''
    Compare actual vs. expected file for a specific test case.
    '''
    flags = open(f'tests/{case}.case').read().split()
    sys.argv = ['test', '-i', '-o', f'tests/{case}.actual'] + flags
    total_space.main(flags=simple_model.flags, model=simple_model.model)
    result = filecmp.cmp(f'tests/{case}.expected', f'tests/{case}.actual')
    if result:
        assert result  # Count successful test assertions
    else:
        pytest.fail(f'The file: tests/{case}.actual '
                    f'is different from the file: tests/{case}.expected')
