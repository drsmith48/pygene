#!/bin/sh

# This 'pre-commit' hook runs pytest on pygene prior to git commits.
# If pytest fails, the commit is aborted.
# To implement this hook, create a symlink to this file in .git/hooks:
#  $ cd .git/hooks
#  $ ln -s ../../pre-commit-hook.sh pre-commit

# pygene tests (run in repo root dir)
echo "The pre-commit hook is running"
pytest --disable-warnings

