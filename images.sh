#!/bin/bash
rm -rf images/*
export PYTHON3=python
export PYTHONPATH=.:$PYTHONPATH
set -eou pipefail
set -x

$PYTHON3 -m total_space.simple_model space -l 'Complete Model' \
| dot -Tsvg > images/complete.svg

$PYTHON3 -m total_space.simple_model -i -P space -l 'Partial Model' \
| dot -Tsvg > images/partial.svg

$PYTHON3 -m total_space.simple_model -i -I space -l 'Invalid Model' \
| dot -Tsvg > images/invalid.svg

$PYTHON3 -m total_space.simple_model -f client-1 space -l 'Focus on Client-1' \
| dot -Tsvg > images/focus.client-1.svg

$PYTHON3 -m total_space.simple_model -f server -C 1 space -l 'Focus on Server (w/ One Client)' \
| dot -Tsvg > images/focus.server.1.svg

$PYTHON3 -m total_space.simple_model -f server space -l 'Focus on Server (w/ Two Clients)' \
| dot -Tsvg > images/focus.server.2.svg

$PYTHON3 -m total_space.simple_model -f server space -c server -l 'Cluster Focus on Server' \
| dot -Tsvg > images/cluster.server.svg

$PYTHON3 -m total_space.simple_model -a -n space -l 'Only Agent State Names' \
| dot -Tsvg > images/agents.svg

$PYTHON3 -m total_space.simple_model -n -C 1 space -m -M -l 'Detail (w/ One Client)' \
| dot -Tsvg > images/detail.1.svg

$PYTHON3 -m total_space.simple_model -n space -m -M -l 'Detail (w/ Two Clients)' \
| dot -Tsvg > images/detail.2.svg

$PYTHON3 -m total_space.simple_model -n -f client-1 space -m -M -l 'Client-1 Detail' \
| dot -Tsvg > images/detail.client-1.svg

$PYTHON3 -m total_space.simple_model -n -f server -C 1 space -m -M -l 'Server Detail (w/ One Client)' \
| dot -Tsvg > images/detail.server.1.svg

$PYTHON3 -m total_space.simple_model -n -f server space -m -M -l 'Server Detail (w/ Two Clients)' \
| dot -Tsvg > images/detail.server.2.svg

$PYTHON3 -m total_space.simple_model -n -f server -i -P space -m -M -l 'Partial Server Detail' \
| dot -Tsvg > images/partial.server.svg

$PYTHON3 -m total_space.simple_model -n -f server -i -I space -m -M -l 'Invalid Server Detail' \
| dot -Tsvg > images/invalid.server.svg

$PYTHON3 -m total_space.simple_model -n -f client-1 -i -P space -m -M -l 'Partial Client-1 Detail' \
| dot -Tsvg > images/partial.client-1.svg

$PYTHON3 -m total_space.simple_model -C 1 time -l 'Client-Server' -c INIT -c 'client-0 @ wait / 2' -c 'client-0 @ idle' \
| dot -Tsvg > images/time.client-server.svg

$PYTHON3 -m total_space.simple_model -i -I time -l 'Invalid' -c INIT -c ' ! ' \
| dot -Tsvg > images/time.invalid.svg

$PYTHON3 -m total_space.simple_model -i -I time -l 'Message Replacement' -c INIT -c '=>' -c INIT \
| dot -Tsvg > images/replace.message.svg
