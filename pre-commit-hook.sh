#!/bin/sh

# This 'pre-commit' hook runs pytest on pygene prior to git commits.
# If pytest fails, the commit is aborted.
# To implement this hook, create to this file in .git/hooks:
#  $ cd .git/hooks
#  $ ln -s ../../pre-commit-hook.sh pre-commit

# pygene tests (run in repo root dir)
pytest --disable-warnings

