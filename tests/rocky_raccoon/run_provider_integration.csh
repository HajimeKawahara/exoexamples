#!/bin/csh -f

if (! -f tests/rocky_raccoon/test_real_column.py) then
    echo "ERROR: run this script from the ExoExamples repository root." >&2
    exit 2
endif

if (! $?EXOGIBBS_REPOSITORY_ROOT) then
    echo "ERROR: set EXOGIBBS_REPOSITORY_ROOT to the ExoGibbs checkout." >&2
    exit 2
endif
if (! -d "$EXOGIBBS_REPOSITORY_ROOT/src/exogibbs") then
    echo "ERROR: invalid ExoGibbs checkout: $EXOGIBBS_REPOSITORY_ROOT" >&2
    exit 2
endif

if (! $?EXOEXAMPLES_JAX_PLATFORM) then
    setenv EXOEXAMPLES_JAX_PLATFORM cpu
endif
switch ("$EXOEXAMPLES_JAX_PLATFORM")
case cpu:
case cuda:
    breaksw
default:
    echo "ERROR: EXOEXAMPLES_JAX_PLATFORM must be cpu or cuda." >&2
    exit 2
endsw

setenv RUN_ROCKY_RACCOON_INTEGRATION 1
setenv JAX_PLATFORMS "$EXOEXAMPLES_JAX_PLATFORM"
setenv JAX_PLATFORM_NAME "$EXOEXAMPLES_JAX_PLATFORM"
setenv JAX_ENABLE_X64 1
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv PYTHONDONTWRITEBYTECODE 1
if ($?PYTHONPATH) then
    setenv PYTHONPATH "$EXOGIBBS_REPOSITORY_ROOT/src:$PYTHONPATH"
else
    setenv PYTHONPATH "$EXOGIBBS_REPOSITORY_ROOT/src"
endif

python -m pytest -p no:cacheprovider -q \
    tests/rocky_raccoon/test_real_column.py \
    tests/rocky_raccoon/test_warm_parent_fixtures.py $argv:q
exit $status
