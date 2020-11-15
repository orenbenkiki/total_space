# Simple Makefile for using this

PYTHON3 = python3  # Needs python 3.5 or above.

all: states pngs complete

states: $(patsubst %_model.py,%_model.states,$(wildcard *_model.py)) \

pngs: $(patsubst %_model.py,%_model.png,$(wildcard *_model.py))

lint: mypy pylint

complete: $(patsubst %_model.py,%_model.complete,$(wildcard *_model.py))

clean:
	rm -f *.yaml *.states *.dot *.png

mypy:
	$(PYTHON3) -m mypy *.py

pylint:
	$(PYTHON3) -m pylint *.py

%_model.yaml: %_model.py
	$(PYTHON3) $? > $@

%_model.states: %_model.yaml
	grep 'name:' $? | sed 's/^\s*name: "//;s/"$$//' > $@

%_model.complete: %_model.states
	if grep -H ' ! ' $<; then false; else true; fi

%.png: %.dot
	dot -Tpng -o $@ $?

%_model.dot: %_model.yaml
	$(PYTHON3) graph.py < $< > $@

# Additional graphs for the simple model:

all: simple_model_by_server.png

simple_model_by_server.dot: simple_model.yaml
	$(PYTHON3) graph.py server < $< > $@
