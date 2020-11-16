#!/bin/sh
rm -rf images/*
export PYTHON3=python3.5
export PYTHONPATH=.:$PYTHONPATH
set -x

$PYTHON3 -m total_space.simple_model dot -l 'Complete Model' \
| dot -Tpng > images/complete.png

$PYTHON3 -m total_space.simple_model -p dot -l 'Partial Model' \
| dot -Tpng > images/partial.png

$PYTHON3 -m total_space.simple_model -i dot -l 'Invalid Model' \
| dot -Tpng > images/invalid.png

$PYTHON3 -m total_space.simple_model -f client-1 dot -l 'Focus on Client-1' \
| dot -Tpng > images/focus.client-1.png

$PYTHON3 -m total_space.simple_model -c 1 -f server dot -l 'Focus on Server (w/ One Client)' \
| dot -Tpng > images/focus.server.1.png

$PYTHON3 -m total_space.simple_model -c 2 -f server dot -l 'Focus on Server (w/ Two Clients)' \
| dot -Tpng > images/focus.server.2.png

$PYTHON3 -m total_space.simple_model -f server dot -c server -l 'Cluster Focus on Server' -c server \
| dot -Tpng > images/cluster.server.png

$PYTHON3 -m total_space.simple_model -a -n dot -l 'Only Agent State Names' \
| dot -Tpng > images/agents.png

$PYTHON3 -m total_space.simple_model -n -f client-1 dot -m -M -l 'Client-1 Detail' \
| dot -Tpng > images/detail.client-1.png

$PYTHON3 -m total_space.simple_model -n -f server dot -m -M -l 'Server Detail' \
| dot -Tpng > images/detail.server.png

$PYTHON3 -m total_space.simple_model -n -f server -p dot -m -M -l 'Partial Server Detail' \
| dot -Tpng > images/partial.server.png

$PYTHON3 -m total_space.simple_model -n -f server -i dot -m -M -l 'Invalid Server Detail' \
| dot -Tpng > images/invalid.server.png

$PYTHON3 -m total_space.simple_model -n -f client-1 -p dot -m -M -l 'Partial Client-1 Detail' \
| dot -Tpng > images/partial.client-1.png
