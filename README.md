# Total Space

Simple code for investigating the total state space of communicating state machines.

## Usage:

Run `make` or `make all` to generate everything. Notable generated files are:

* `<something>_model.states` - lists all total configuration states, one per line.
  You can use `grep` to verify model properties.
  For example, there should be no lines containing `MISSING LOGIC`.
  However this can be extended to testing invariants (e.g., if agent A is in state X then agent is never in state Y).

* `<something>_model.png` - A diagram of all the total configuration states and the transitions between them.
  You can ask for the graph to be clustered by the states of one or more agents.
  For example the `Makefile` will create `simple_model_by_server.png` which is clustered by the server state.

Run `make clean` to remove all the generated files.

NOTE: The simple model provided here is partial, so `make` will fail (to demonstrate this functionality).
You can still view the diagrams (which help in understanding where the model incompleteness is).
To complete the model, uncomment the annotated lines in the `simple_model.py`.

## Extend:

1. Create a model file named `<something>_model.py`.
   See `simple_model.py` for an example, and `total_space.py` for the (sparse) documentation.

2. Run `make lint` to perform basic sanity checking of your code.

3. Run `make complete` to verify the model is complete.

4. Run `make pngs` to visualize the model.
   Optionally, add rules for additional (clustered) graphs in the `Makefile`.

## TODO:

0. This should be placed in a git repository somewhere.

1. The generated graph for the total system configuration space quickly becomes unwieldy.
   There should be a way to generate graphs that focus on the state of a single agent, ignoring the rest.

2. There should be code for parsing the states file into structured data, to make it easier to express and verify invariants.

3. The code should be ported to python 3.6 to use the friendlier syntax for named tuples.

4. Since the code encourages using tuples instead of lists, methods such as `_replace` should be exported.
   In principle such methods should have been available by python's standard library.
