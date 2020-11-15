Total Space
===========

Investigate the total state space of communicating finite state machines.

Specifically:

Given a model of a system comprising of
multiple agents,
where each agent is a non-deterministic state machine,
which responds to either time or receiving a message with one of some possible actions,
where each such action can change the agent state and/or send messages to other agents;
Then this package will generate the total possible state space of the overall system,
validate the model for completeness,
validate each system state for additional arbitrary correctness criteria,
and visualize the states and transitions in various ways.

Requirements
------------

Python 3.5 or above.

Installation
------------

Run ``pip3 --install total_space``.

Usage
-----

Create a ``model.py`` file that:

* Imports the ``total_space`` module.

* Declares state and agent classes for modeling some system.

* Provides functions for:

    * (Optionally) add command line flags to control the model creation.

    * Create a function that creates the model given the parsed command line flags.

    * (Optionally) report invalid conditions in a global system configuration.

* Invokes the ``total_space.main`` function, passing it the above.

You can now run ``python3 model.py`` to do the following:

* Generate a list of all the possible system configuration states.
  Each system configuration state contains the state of all the agents,
  the messages that are in flight, and invalid conditions (if any).
  A complete and correct model would contain no invalid conditions.

* Generates a tab-separated file of all the transitions between the possible system configuration states.
  The code doesn't generate any transitions from an invalid system configuration state.

* Generate a ``dot`` file visualizing the transitions between the system configuration states.
  Invalid system configuration states are highlighted.

* Focus only on the states of specific agent(s).

Run ``python3 model.py -h`` for details.
