#!/bin/bash
rm -rf images/*
export PYTHON3=python
export PYTHONPATH=.:$PYTHONPATH
set -eou pipefail
set -x

$PYTHON3 -m total_space.simple_model space -l 'Complete Model' \
| dot -Tpng > images/complete.png

$PYTHON3 -m total_space.simple_model -p space -l 'Partial Model' \
| dot -Tpng > images/partial.png

$PYTHON3 -m total_space.simple_model -i space -l 'Invalid Model' \
| dot -Tpng > images/invalid.png

$PYTHON3 -m total_space.simple_model -f client-1 space -l 'Focus on Client-1' \
| dot -Tpng > images/focus.client-1.png

$PYTHON3 -m total_space.simple_model -f server -c 1 space -l 'Focus on Server (w/ One Client)' \
| dot -Tpng > images/focus.server.1.png

$PYTHON3 -m total_space.simple_model -f server space -l 'Focus on Server (w/ Two Clients)' \
| dot -Tpng > images/focus.server.2.png

$PYTHON3 -m total_space.simple_model -f server space -c server -l 'Cluster Focus on Server' \
| dot -Tpng > images/cluster.server.png

$PYTHON3 -m total_space.simple_model -a -n space -l 'Only Agent State Names' \
| dot -Tpng > images/agents.png

$PYTHON3 -m total_space.simple_model -n -c 1 space -m -M -l 'Detail (w/ One Client)' \
| dot -Tpng > images/detail.1.png

$PYTHON3 -m total_space.simple_model -n space -m -M -l 'Detail (w/ Two Clients)' \
| dot -Tpng > images/detail.2.png

$PYTHON3 -m total_space.simple_model -n -f client-1 space -m -M -l 'Client-1 Detail' \
| dot -Tpng > images/detail.client-1.png

$PYTHON3 -m total_space.simple_model -n -f server -c 1 space -m -M -l 'Server Detail (w/ One Client)' \
| dot -Tpng > images/detail.server.1.png

$PYTHON3 -m total_space.simple_model -n -f server space -m -M -l 'Server Detail (w/ Two Clients)' \
| dot -Tpng > images/detail.server.2.png

$PYTHON3 -m total_space.simple_model -n -f server -p space -m -M -l 'Partial Server Detail' \
| dot -Tpng > images/partial.server.png

$PYTHON3 -m total_space.simple_model -n -f server -i space -m -M -l 'Invalid Server Detail' \
| dot -Tpng > images/invalid.server.png

$PYTHON3 -m total_space.simple_model -n -f client-1 -p space -m -M -l 'Partial Client-1 Detail' \
| dot -Tpng > images/partial.client-1.png

$PYTHON3 -m total_space.simple_model -c 1 time -c INIT -c 'client-0 @ wait' -c 'client-0 @ idle' \
| dot -Tpng > images/time.client-server.png

$PYTHON3 -m total_space.simple_model -i time -c INIT -c ' ! ' \
| dot -Tpng > images/time.invalid.png
