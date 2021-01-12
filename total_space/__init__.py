'''
Investigate the total state space of communicating state machines.
'''

# pylint_ disable=C0330
# pylint: disable=inherit-non-class
# pylint: disable=len-as-condition
# pylint: disable=line-too-long
# pylint: disable=multiple-statements
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=pointless-statement
# pylint: disable=protected-access
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-lines
# pylint: disable=unsubscriptable-object
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import


from argparse import ArgumentParser
from argparse import Namespace
from copy import copy
import re
import sys
from contextlib import contextmanager
from functools import total_ordering
from typing import *


__version__ = '0.2.5'


__all__ = [
    'Immutable',
    'initializing',
    'State',
    'Message',
    'Action',
    'Agent',
    'Invalid',
    'Configuration',
    'Validation',
    'Transition',
    'System',
    'main',
]


@total_ordering
class Immutable:
    '''
    Prevent any properties from being modified.

    The code here is in a "functional" style and makes heavy use of immutable
    data.
    '''

    #: Allow modification of properties when initializing the object.
    _is_initializing = False

    def __setattr__(self, name: str, value: Any) -> None:
        if Immutable._is_initializing:
            object.__setattr__(self, name, value)
        else:
            raise RuntimeError('trying to modify the property: %s of an immutable: %s'
                               % (name, self.__class__.__qualname__))

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __lt__(self, other) -> bool:
        return str(self) < str(other)


@contextmanager
def initializing() -> Iterator[None]:
    '''
    Allow mutating :py:const:`Immutable` data during initialization.
    '''
    try:
        was_initalizing = Immutable._is_initializing
        Immutable._is_initializing = True
        yield
    finally:
        Immutable._is_initializing = was_initalizing


class State(Immutable):
    '''
    A state of an :py:const:`Agent`.

    In general the state of an agent could be "anything" (including "nothing" -
    None).  It "should" be immutable; use named tuples for structures and
    simple tuples for arrays/lists.

    In general one can bury all the information in the data field and use a
    single state name.  However for clarity, the overall space of all possible
    states is split into named sub-spaces.
    '''

    __slots__ = ['name', 'data']

    def __init__(self, *, name: str, data: Any = None) -> None:
        with initializing():
            #: The state's name, used for visualization and method names.
            self.name = name

            #: The state's actual data.
            self.data = data

    def validate(self) -> Collection[str]:  # pylint: disable=no-self-use
        '''
        Return a hopefully empty collection of reasons the state is invalid.
        '''
        return ()

    def __str__(self) -> str:
        if self.data in [None, (), [], '', set(), {}]:
            return self.name
        return '%s / %s' % (self.name, self.data)

    def only_names(self) -> 'State':
        '''
        Remove the data for a simplified view of the message.
        '''
        return State(name=self.name, data=None)


class Message(Immutable):
    '''
    A message sent from one :py:const:`Agent` to another (or a time message).

    This has the same name-vs-data considerations as the agent
    :py:const:`State` above.
    '''

    __slots__ = ['source_agent_name', 'target_agent_name', 'state']

    def __init__(self, *, source_agent_name: str, target_agent_name: str, state: State) -> None:
        with initializing():
            #: The name of the agent that generated the message, or ``time``.
            self.source_agent_name = source_agent_name

            #: The name of the agent that will receive the message.
            self.target_agent_name = target_agent_name

            #: The state carried by the message.
            self.state = state

    @staticmethod
    def time(agent: 'Agent') -> 'Message':
        '''
        We shoehorn the time-passes event to look like a message whose source
        is the agent state name, whose name is ``time``, and whose data is
        empty (``None``).
        '''
        return Message(source_agent_name='@ %s' % agent.state.name, target_agent_name=agent.name, state=State(name='time'))

    def __str__(self) -> str:
        return '%s -> %s -> %s' % (self.source_agent_name, self.state, self.target_agent_name)

    def validate(self) -> Collection[str]:  # pylint: disable=no-self-use
        '''
        Return a hopefully empty collection of reasons the message is invalid.

        This is only invoked if the state is valid as of itself.
        '''
        return ()

    def only_names(self) -> 'Message':
        '''
        Remove the data for a simplified view of the message.
        '''
        return Message(source_agent_name=self.source_agent_name,
                       target_agent_name=self.target_agent_name,
                       state=State(name=self.state.name))


class Action(Immutable):
    '''
    A possible action taken by an :py:const:`Agent` as a response to receiving
    a :py:const:`Message`.
    '''

    __slots__ = ['name', 'next_state', 'send_messages']

    #: An action that does not change the state or send any messages.
    NOP: 'Action'

    def __init__(self, *, name: str, next_state: Optional[State] = None, send_messages: Collection[Message] = ()) -> None:
        with initializing():
            #: The name of the action for visualization.
            self.name = name

            #: The next :py:const:`State` of the agent. If ``None`` the agent remains in the same state.
            self.next_state = next_state

            #: Any :py:const:`Message` sent by the agent.
            #: We assume the communication fabric may reorder messages.
            #: That is, there's no guarantee at what order the sent messages will be received by their targets.
            self.send_messages = tuple(send_messages)

Action.NOP = Action(name='nop')


class Agent(Immutable):
    '''
    An agent in the :py:const:`Configuration`.

    Each agent is a non-deterministic state machine.

    Sub-classes should implement methods with the name
    ``_<message_name>_when_<state_name>``, which are invoked when the has a
    :py:const:`State` with ``state_name``, and receives a :py:const:`Message`
    with the name ``message_name``.

    Each such method should have a single parameter for the message, and return
    either ``None`` or a ``Collection`` of possible :py:const:`Action`, any one
    of which may be taken when receiving the message while at the current
    state.  Providing multiple alternatives allows modeling a non-deterministic
    agent (e.g., modeling either a hit or a miss in a cache).

    If the method returns :py:attr:`Agent.IGNORE`, then the agent silently
    ignores the message without changing its state. This is a convenient
    shorthand for returning a collection with a single :py:attr:`Action.NOP`
    action.

    A common pattern is for an agent to contain serial sub-flows, where upon
    receiving some message, it emits a request and enters a state where it
    awaits the response, and refuses to receive any other message until it
    arrives; any messages arriving during this window are deferred until the
    response arrives and are only handled then.

    To implement this, the agent should override the
    :py:func:`Agent.is_deferring` method to return ``True`` while in the
    window. When this is the case, you only need to implement handler methods
    for the acceptable messages; messages for which no handler method exists
    are automatically considered to be deferred. This is equivalent to creating
    explicit handler methods that return :py:attr:`Agent.DEFER`.

    Once a response message that ends the window arrives, the handler method
    should return an action that moves the agent to a different state, in which
    :py:func:`Agent.is_deferring` will return ``False``.
    '''

    __slots__ = ['name', 'state']

    #: Return this from a handler to indicate the agent silently ignores the message.
    IGNORE = (Action.NOP,)

    #: Return this from a handler to indicate the agent defers handling the message
    #: until it is in a different state.
    DEFER = ()

    #: Return this from a handler to indicate the agent is not expected to even
    #: receive some message while in some state, or does not know how to handle
    #: some message while in some state. If this happens, the model is either
    #: wrong or partial, a problem will be reported, and the model will need to
    #: be fixed.
    UNEXPECTED = None

    def __init__(self, *, name: str, state: State) -> None:
        with initializing():
            #: The name of the agent for visualization.
            self.name = name

            #: The state of the agent.
            self.state = state

    def is_deferring(self) -> bool:  # pylint: disable=no-self-use
        '''
        Return whether the agent will be deferring some messages while in the
        current state.

        If this is true, then the agent is inside some "interrupt window",
        basically blocking until a specific message arrives to complete the
        window and move to a different (normal) state. While in this window,
        missing handler methods are treated as if they return
        :py:attr:`Agent.DEFER`, indicating the message would only be handled
        following (and not during) the window
        '''
        return False

    def with_state(self, state: State) -> 'Agent':
        '''
        Return a new agent with a modified state.
        '''
        with initializing():
            other = copy(self)
            other.state = state
        return other

    def validate(self) -> Collection[str]:  # pylint: disable=no-self-use
        '''
        Return a hopefully empty collection of reasons the agent is invalid.

        This is only invoked if the state is valid as of itself.
        '''
        return ()

    def __str__(self) -> str:
        name = '%s @ %s' % (self.name, self.state.name)
        if self.state.data not in [None, (), [], '']:
            name += ' / %s' % (self.state.data,)
        return name

    def only_names(self) -> 'Agent':
        '''
        Remove the data for a simplified view of the message.
        '''
        return Agent(name=self.name, state=self.state.only_names())

    @staticmethod
    def no_action() -> Action:
        '''
        Return an action that doesn't do anything (keeps the state and sends no
        messages).
        '''
        return Action(name='nop')


class Invalid(Immutable):
    '''
    Indicate something is invalid.
    '''

    __slots__ = ['kind', 'name', 'reason']

    def __init__(self, *, kind: str, name: Optional[str] = None, reason: str) -> None:
        assert kind in ['agent', 'message', 'configuration']
        assert (name is None) == (kind == 'configuration')

        with initializing():
            #: The kind of invalid condition (``agent``, ``message``, or a whole system ``configuration``).
            self.kind = kind

            #: The name of whatever is invalid.
            self.name = name

            #: The reason for the invalid condition (short one line text).
            self.reason = reason

    def __str__(self) -> str:
        if self.name is None:
            return '%s is invalid because: %s' % (self.kind, self.reason)
        return '%s: %s is invalid because: %s' % (self.kind, self.name, self.reason)


class Configuration(Immutable):
    '''
    The total configuration of the whole system.
    '''

    __slots__ = ['name', 'agents', 'messages_in_flight', 'invalids']

    def __init__(
        self,
        *,
        agents: Collection[Agent],
        messages_in_flight: Collection[Message] = (),
        invalids: Collection[Invalid] = ()
    ) -> None:
        with initializing():
            #: All the agents with their state.
            self.agents = tuple(sorted(agents))

            #: The messages in-flight between agents.
            self.messages_in_flight = tuple(sorted(messages_in_flight))

            #: Everything invalid in this configuration.
            self.invalids = tuple(sorted(invalids))

            name = ' , '.join([str(agent) for agent in self.agents])
            if len(self.messages_in_flight) > 0:
                name += ' ; '
                name += ' , '.join([str(message) for message in self.messages_in_flight])
            if len(self.invalids) > 0:
                name += ' ! '
                name += ' , '.join([str(invalid) for invalid in self.invalids])

            #: A name fully describing the total configuration.
            self.name = name

    def __str__(self) -> str:
        return self.name

    @property
    def valid(self) -> bool:
        '''
        Report whether the configuration is valid.
        '''
        return len(self.invalids) == 0

    def focus_on_agents(self, agent_indices: Collection[int]) -> 'Configuration':
        '''
        Return a simplified configuration focusing only on some of the agents.
        '''
        agents = tuple([self.agents[agent_index] for agent_index in agent_indices])
        agent_names = {agent.name for agent in agents}
        messages_in_flight = tuple([message for message in self.messages_in_flight
                                    if message.source_agent_name in agent_names or message.target_agent_name in agent_names])
        invalids = self.invalids
        return Configuration(agents=agents, messages_in_flight=messages_in_flight, invalids=invalids)

    def only_names(self) -> 'Configuration':
        '''
        Remove all the data, keeping the names, for a simplified view of the
        configuration.
        '''
        agents = tuple(agent.only_names() for agent in self.agents)
        messages_in_flight = tuple(message.only_names() for message in self.messages_in_flight)
        invalids = self.invalids
        return Configuration(agents=agents, messages_in_flight=messages_in_flight, invalids=invalids)

    def only_agents(self) -> 'Configuration':
        '''
        Remove all the in-flight messages, keeping the agents, for a simplified
        view of the configuration.
        '''
        return Configuration(agents=self.agents, messages_in_flight=(), invalids=self.invalids)


class Transition(Immutable):
    '''
    A transition between one :py:const:`Configuration` to another.
    '''

    __slots__ = ['from_configuration_name', 'delivered_message', 'to_configuration_name']

    def __init__(self, *, from_configuration_name: str, delivered_message: Message, to_configuration_name: str) -> None:
        with initializing():
            #: The name of the configuration before the transition.
            self.from_configuration_name = from_configuration_name

            #: The message that was delivered to an agent to trigger the transition.
            self.delivered_message = delivered_message

            #: The name of the configuration after the transition.
            self.to_configuration_name = to_configuration_name

    def __str__(self) -> str:
        return '%s => %s => %s' % (self.from_configuration_name, self.delivered_message, self.to_configuration_name)


#: The type of a function that validates a :py:const:`Configuration`,
#: returning a hopefully empty collection of reasons it is invalid.
Validation = Callable[[Configuration], Collection[str]]

class System(Immutable):
    '''
    The total state space of the whole system.
    '''

    __slots__ = ['configurations', 'transitions']

    def __init__(
        self,
        *,
        configurations: Tuple[Configuration, ...],
        transitions: Tuple[Transition, ...]
    ) -> None:
        with initializing():
            #: All the possible configurations the system could be at.
            self.configurations = configurations

            #: All the transitions between the configurations.
            self.transitions = transitions

    @staticmethod
    def compute(
        *,
        agents: Collection[Agent],
        validate: Optional[Validation] = None
    ) -> 'System':
        '''
        Compute the total state space of a system given some agents in their
        initial state.
        '''
        model = Model(agents, validate)
        return System(configurations=tuple(sorted(model.configurations.values())),
                      transitions=tuple(sorted(model.transitions)))

    def focus_on_agents(self, keep_agents: Collection[str]) -> 'System':
        '''
        Return a simplified view of the system which focuses a subset of the
        agents.
        '''
        assert len(keep_agents) > 0
        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(self.configurations[0].agents)}
        keep_indices = sorted([agent_indices[agent_name] for agent_name in keep_agents])

        return self.simplify(lambda configuration: configuration.focus_on_agents(keep_indices),
                             lambda message: message)

    def only_names(self) -> 'System':
        '''
        Strip all the data, keeping only the names, for a simpler view of the
        system.
        '''
        return self.simplify(lambda configuration: configuration.only_names(),
                             lambda message: message.only_names())

    def only_agents(self) -> 'System':
        '''
        Strip all the in-flight messages, keeping only the agents, for a
        simpler view of the system.
        '''
        return self.simplify(lambda configuration: configuration.only_agents(),
                             lambda message: message)

    def simplify(self,
                 simplify_configuration: Callable[[Configuration], Configuration],
                 simplify_message: Callable[[Message], Message]) -> 'System':
        '''
        Return a simplified view of the system.
        '''
        new_configuration_by_name: Dict[str, Configuration] = {}
        new_name_by_old_name: Dict[str, str] = {}

        for configuration in self.configurations:
            new_configuration = simplify_configuration(configuration)
            new_configuration_by_name[new_configuration.name] = new_configuration
            new_name_by_old_name[configuration.name] = new_configuration.name

        new_transitions: Dict[str, Transition] = {}
        reachable_configuration_names: Set[str] = set()
        for transition in self.transitions:
            new_from_configuration_name = new_name_by_old_name[transition.from_configuration_name]
            new_to_configuration_name = new_name_by_old_name[transition.to_configuration_name]
            if new_from_configuration_name == new_to_configuration_name:
                continue
            reachable_configuration_names.add(new_from_configuration_name)
            reachable_configuration_names.add(new_to_configuration_name)
            new_transition = Transition(from_configuration_name=new_from_configuration_name,
                                        delivered_message=simplify_message(transition.delivered_message),
                                        to_configuration_name=new_to_configuration_name)
            new_transitions[str(new_transition)] = new_transition

        new_configurations = tuple([new_configuration_by_name[name] for name in sorted(reachable_configuration_names)])
        return System(configurations=new_configurations, transitions=tuple(sorted(new_transitions.values())))

    def print_states(self, file: 'TextIO') -> None:
        '''
        Print a list of all the system configurations to a file.
        '''
        for configuration in self.configurations:
            file.write('%s\n' % configuration.name)

    def print_transitions(self, file: 'TextIO', path: List[str], sent_messages: bool) -> None:
        '''
        Print a list of all the transitions between system configurations to a
        tab-separated file.
        '''
        if len(path) > 0:
            transitions: Collection[Transition] = self.transitions_path(path)
        else:
            transitions = self.transitions

        configuration_by_name = {configuration.name: configuration for configuration in self.configurations}

        file.write('from_configuration_name\t')
        file.write('delivered_message_source_agent_name\t')
        file.write('delivered_message_name\t')
        file.write('delivered_message_data\t')
        file.write('delivered_message_target_agent_name\t')
        if sent_messages:
            file.write('sent_messages\t')
        file.write('to_configuration_name\n')

        for transition in transitions:
            from_configuration_name = transition.from_configuration_name
            to_configuration_name = transition.to_configuration_name
            if sent_messages:
                from_configuration = configuration_by_name[from_configuration_name]
                to_configuration = configuration_by_name[to_configuration_name]
                from_configuration_name = from_configuration.only_agents().name
                to_configuration_name = to_configuration.only_agents().name

            file.write('%s\t' % from_configuration_name)
            file.write('%s\t' % transition.delivered_message.source_agent_name)
            file.write('%s\t' % transition.delivered_message.state.name)
            file.write('%s\t' % transition.delivered_message.state.data)
            file.write('%s\t' % transition.delivered_message.target_agent_name)

            if sent_messages:
                messages = new_messages(from_configuration, to_configuration)
                if len(messages) > 0:
                    file.write(' , '.join([str(message) for message in messages]))
                    file.write('\t')
                else:
                    file.write('None\t')

            file.write('%s\n' % to_configuration_name)

    def transitions_path(self, paths: List[str]) -> List[Transition]:
        '''
        Return the path of transitions between configurations matching the
        patterns.
        '''
        assert len(paths) > 1

        initial_configuration_names = self.matching_configuration_names(paths[0])
        if len(initial_configuration_names) != 1:
            raise ValueError('first regexp pattern: %s matches more than one configuration' % paths[0])

        configuration_name = initial_configuration_names[0]

        outgoing_transitions: Dict[str, List[Transition]] = {}
        for transition in self.transitions:
            transitions_list = outgoing_transitions.get(transition.from_configuration_name)
            if transitions_list is None:
                transitions_list = outgoing_transitions[transition.from_configuration_name] = []
            transitions_list.append(transition)

        transitions: List[Transition] = []
        for pattern in paths[1:]:
            self.shortest_path(configuration_name, pattern, outgoing_transitions, transitions)
            configuration_name = transitions[-1].to_configuration_name

        return transitions

    def shortest_path(
        self,
        from_configuration_name: str,
        to_pattern: str,
        outgoing_transitions: Dict[str, List[Transition]],
        transitions: List[Transition]
    ) -> None:
        '''
        Return the shortest path from a specific configuration to a
        configuration that matches the specified pattern.
        '''
        to_configuration_names = set(self.matching_configuration_names(to_pattern))

        near_pending: List[Tuple[str, List[Transition]]] = [(from_configuration_name, [])]
        far_pending: List[Tuple[str, List[Transition]]] = []
        visited_configuration_names: Set[str] = set()
        while len(near_pending) > 0:
            configuration_name, near_transitions = near_pending.pop()
            if configuration_name not in visited_configuration_names:
                visited_configuration_names.add(configuration_name)

                for transition in outgoing_transitions[configuration_name]:
                    far_transitions = near_transitions + [transition]
                    if transition.to_configuration_name in to_configuration_names:
                        transitions.extend(far_transitions)
                        return

                    far_pending.append((transition.to_configuration_name, far_transitions))

            if len(near_pending) == 0:
                near_pending = far_pending
                far_pending = []

        raise RuntimeError('there is no path from the configuration: %s to a configuration matching the pattern: %s'
                           % (from_configuration_name, to_pattern))

    def matching_configuration_names(self, pattern: str) -> List[str]:
        '''
        Return all the names of the configurations that match a pattern.
        '''
        try:
            regexp = re.compile(pattern)
        except BaseException:
            raise ValueError('invalid regexp pattern: %s' % pattern)  # pylint: disable=raise-missing-from

        configuration_names = [configuration.name for configuration in self.configurations if regexp.search(configuration.name)]
        if len(configuration_names) == 0:
            raise ValueError('the regexp pattern: %s does not match any configurations' % pattern)

        return configuration_names

    def print_space(
        self,
        file: 'TextIO',
        *,
        cluster_by_agents: Collection[str] = (),
        label: str = 'Total Space',
        separate_messages: bool = False,
        merge_messages: bool = False
    ) -> None:
        '''
        Print a ``dot`` file visualizing the space of the possible system
        configuration states and the transitions between them.
        '''
        file.write('digraph G {\n')
        file.write('fontname = "Sans-Serif";\n')
        file.write('fontsize = 32;\n')
        file.write('node [ fontname = "Sans-Serif" ];\n')
        file.write('edge [ fontname = "Sans-Serif" ];\n')
        file.write('label = "%s";\n' % label)

        reachable_configuration_names: Set[str] = set()
        self.print_space_edges(file, reachable_configuration_names,
                               separate_messages=separate_messages, merge_messages=merge_messages)
        self.print_space_nodes(file, cluster_by_agents, reachable_configuration_names, merge_messages)

        file.write('}\n')

    def print_space_nodes(  # pylint: disable=too-many-locals
        self,
        file: 'TextIO',
        cluster_by_agents: Collection[str],
        reachable_configuration_names: Set[str],
        merge_messages: bool
    ) -> None:
        '''
        Print all the nodes of the ``dot`` file.
        '''
        node_names: Set[str] = set()
        if merge_messages:
            configurations = tuple([configuration.only_agents() for configuration in self.configurations])
        else:
            configurations = self.configurations

        if len(cluster_by_agents) == 0:
            for configuration in configurations:
                if configuration.name in reachable_configuration_names:
                    print_space_node(file, configuration, node_names)
            return

        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(configurations[0].agents)}
        cluster_by_indices = [agent_indices[agent_name] for agent_name in cluster_by_agents]
        paths = [['%s @ %s' % (configuration.agents[agent_index].name, configuration.agents[agent_index].state.name)
                  for agent_index in cluster_by_indices]
                 for configuration in configurations
                 if configuration.name in reachable_configuration_names]

        current_path: List[str] = []
        for path, configuration in sorted(zip(paths, configurations)):
            remaining_path = current_path + []
            while len(remaining_path) > 0 and len(current_path) > 0 and remaining_path[0] == path[0]:
                remaining_path = remaining_path[1:]
                path = path[1:]

            while len(remaining_path) > 0:
                remaining_path.pop()
                current_path.pop()
                file.write('}\n')

            for cluster in path:
                current_path.append(cluster)
                file.write('subgraph "cluster_%s" {\n' % ' , '.join(current_path))
                file.write('fontsize = 24;\n')
                file.write('label = "%s";\n' % cluster)

            print_space_node(file, configuration, node_names)

        while len(current_path) > 0:
            current_path.pop()
            file.write('}\n')

    def print_space_edges(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        file: 'TextIO',
        reachable_configuration_names: Set[str],
        *,
        separate_messages: bool,
        merge_messages: bool
    ) -> None:
        '''
        Print all the edges of the ``dot`` file.
        '''
        configuration_by_name = {configuration.name: configuration for configuration in self.configurations}
        agent_names = {agent.name for agent in list(configuration_by_name.values())[0].agents}

        message_edges: Set[str] = set()
        message_nodes: Set[str] = set()
        intermediate_nodes: Set[str] = set()

        for transition in self.transitions:
            delivered_message: Message = transition.delivered_message

            shown_source = delivered_message.source_agent_name in agent_names
            known_source = delivered_message.state.name == 'time' or shown_source
            known_target = delivered_message.target_agent_name in agent_names

            if not known_source and not known_target:
                continue

            from_configuration = configuration_by_name[transition.from_configuration_name]
            to_configuration = configuration_by_name[transition.to_configuration_name]

            sent_message: Optional[Message] = None
            if separate_messages:
                if len(to_configuration.messages_in_flight) > len(from_configuration.messages_in_flight):
                    assert len(to_configuration.messages_in_flight) == len(from_configuration.messages_in_flight) + 1
                    for message in new_messages(from_configuration, to_configuration):
                        if (message.source_agent_name in agent_names or message.target_agent_name in agent_names):
                            sent_message = message
                            break

            if merge_messages:
                from_configuration = from_configuration.only_agents()
                to_configuration = to_configuration.only_agents()

            if not shown_source and not known_target and from_configuration.name == to_configuration.name:
                continue

            if not separate_messages:
                reachable_configuration_names.add(from_configuration.name)
                reachable_configuration_names.add(to_configuration.name)
                file.write('"%s" -> "%s" [ penwidth=3, label="%s" ];\n'
                           % (from_configuration.name,
                              to_configuration.name,
                              message_space_label(delivered_message).replace(' | ', '\n')))
                continue

            edges: List[str] = []

            if sent_message is None:
                intermediate = '%s => %s' % (from_configuration.name, to_configuration.name)
            else:
                intermediate = '%s => %s => %s' \
                    % (from_configuration.name, delivered_message, to_configuration.name)

            if sent_message is not None:
                print_space_message(file, sent_message, message_nodes)
                edges.append('"%s" -> "%s" [ penwidth=3, color=mediumblue ];\n' % (intermediate, message_space_label(sent_message)))
                arrowhead = 'none'
            else:
                arrowhead = 'normal'

            if known_target or sent_message is not None:
                print_space_message(file, delivered_message, message_nodes)
                edges.append('"%s" -> "%s" [ penwidth=3, color=mediumblue, dir=forward, arrowhead=%s ];\n'
                             % (message_space_label(delivered_message), intermediate, arrowhead))

            assert from_configuration.valid
            if to_configuration.valid:
                color = 'darkgreen'
            else:
                color = 'crimson'

            if len(edges) == 0:
                continue

            reachable_configuration_names.add(from_configuration.name)
            reachable_configuration_names.add(to_configuration.name)

            edges += [
                '"%s" -> "%s" [ penwidth=3, color=%s, dir=forward, arrowhead=none ];\n'
                    % (from_configuration.name, intermediate, color),
                '"%s" -> "%s" [ penwidth=3, color=%s ];\n'
                    % (intermediate, to_configuration.name, color)
            ]

            if intermediate not in intermediate_nodes:
                file.write('"%s" [ shape=box, label="", penwidth=4, width=0, height=0, color="#0063cd" ];\n' % intermediate)
                intermediate_nodes.add(intermediate)

            for edge in edges:
                if edge not in message_edges:
                    file.write(edge)
                    message_edges.add(edge)

    def print_time(self, file: 'TextIO', label: str, path: List[str]) -> None:  # pylint: disable=too-many-locals
        '''
        Print a ``dot`` file visualizing the interaction between agents along
        the specified path.
        '''
        transitions = self.transitions_path(path)

        configuration_by_name = {configuration.name: configuration for configuration in self.configurations}
        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(self.configurations[0].agents)}

        message_id_by_times, message_lifetime_by_id = message_times(transitions, configuration_by_name)

        file.write('digraph G {\n')
        file.write('fontname = "Sans-Serif";\n')
        file.write('fontsize = 32;\n')
        file.write('node [ fontname = "Sans-Serif" ];\n')
        file.write('edge [ fontname = "Sans-Serif" ];\n')
        file.write('label = "%s";\n' % label)
        file.write('ranksep = 0.05;\n')

        final_configuration = configuration_by_name[transitions[-1].to_configuration_name]
        for invalid in final_configuration.invalids:
            print_invalid_time_node(file, invalid)

        printed_messages: Set[int] = set()

        for agent in self.configurations[0].agents:
            last_agent_node, last_message_name, last_message_node = \
                print_agent_time_nodes(file, transitions, configuration_by_name, message_id_by_times,
                                       agent.name, agent_indices[agent.name])

            for message_id, (message, first_time, last_time) in message_lifetime_by_id.items():
                if message.source_agent_name == agent.name:
                    print_message_time_nodes(file, message_id, message, first_time, last_time)
                    printed_messages.add(message_id)

            file.write('}\n')

            for invalid in final_configuration.invalids:
                if invalid.kind == 'agent' and invalid.name == agent.name:
                    file.write('"%s" -> "%s" [ penwidth=3, color=crimson, weight=1000 ];\n' % (last_agent_node, str(invalid)))
                elif invalid.kind == 'message' and invalid.name == last_message_name:
                    file.write('"%s" -> "%s" [ penwidth=3, color=crimson, weight=1000 ];\n' % (last_message_node, str(invalid)))
                else:
                    file.write('"%s" -> "%s" [ style=invis ];\n' % (last_agent_node, str(invalid)))

        for message_id, (message, first_time, last_time) in message_lifetime_by_id.items():
            if message_id not in printed_messages:
                print_message_time_nodes(file, message_id, message, first_time, last_time)

        file.write('}\n')


def new_messages(from_configuration: Configuration, to_configuration: Configuration) -> List[Message]:
    '''
    Return all the messages that exist in one configuration but not the other.
    '''
    return [message
            for message in to_configuration.messages_in_flight
            if message not in from_configuration.messages_in_flight]


def print_space_node(file: 'TextIO', configuration: Configuration, node_names: Set[str]) -> None:
    '''
    Print a node for a system configuration state.
    '''
    if configuration.name in node_names:
        return
    node_names.add(configuration.name)

    if configuration.valid:
        color = 'palegreen'
        label = configuration.name.replace(' , ', '\n').replace(' ; ', '\n')
    else:
        color = 'lightcoral'
        label = '\n\n'.join([invalid_label(invalid) for invalid in configuration.invalids])
    file.write('"%s" [ label="%s", shape=box, style=filled, color=%s];\n' % (configuration.name, label, color))


def print_space_message(file: 'TextIO', message: Message, message_nodes: Set[str]) -> None:
    '''
    Print a node for an in-flight messages.
    '''
    label = message_space_label(message)
    if label in message_nodes:
        return
    message_nodes.add(label)
    file.write('"%s" [ label="{%s}", shape=record, style=filled, color=paleturquoise ];\n' % (label, label))


def message_space_label(message: Message) -> str:
    '''
    The label to show for a message.
    '''
    return '%s &rarr; | %s | &rarr; %s' % (message.source_agent_name, message.state, message.target_agent_name)


def invalid_label(invalid: Invalid) -> str:
    '''
    The label to show for an invalid notification.
    '''
    label = str(invalid)
    label = label.replace(' because:', '\nbecause:')
    label = label.replace(' for message:', '\nfor message:')
    label = label.replace(' when in state:', '\nwhen in state:')
    return label


def message_times(  # pylint: disable=too-many-locals
    transitions: List[Transition],
    configuration_by_name: Dict[str, Configuration]
) -> Tuple[Dict[Tuple[int, str], int], Dict[int, Tuple[Message, int, int]]]:
    '''
    Return for each time and message the unique message id, and for each unique
    message id the message and the first and last time it existed.
    '''
    message_id_by_times: Dict[Tuple[int, str], int] = {}
    message_lifetime_by_id: Dict[int, Tuple[Message, int, int]] = {}
    active_messages: Dict[str, Tuple[Message, int, int]] = {}

    time_counter = 1
    configuration = configuration_by_name[transitions[0].from_configuration_name]

    for message in configuration.messages_in_flight:
        message_text = str(message)
        if message_text in active_messages.keys():
            raise NotImplementedError('multiple instances of the same message: %s' % message_text)
        message_id = len(active_messages)
        message_id_by_times[(0, message_text)] = message_id
        active_messages[message_text] = (message, message_id, 0)

    next_message_id = len(active_messages)

    for transition in transitions:
        to_configuration = configuration_by_name[transition.to_configuration_name]
        mid_time_counter = time_counter + 1
        to_time_counter = time_counter + 2
        to_active_messages: Dict[str, Tuple[Message, int, int]] = {}

        for message in to_configuration.messages_in_flight:
            message_text = str(message)
            if message_text in to_active_messages:
                raise NotImplementedError('multiple instances of the same message: %s' % message_text)

            active = active_messages.get(message_text)
            if active is None:
                active = (message, next_message_id, to_time_counter)
                next_message_id += 1
            to_active_messages[message_text] = active
            message_id_by_times[(mid_time_counter, message_text)] = active[1]
            message_id_by_times[(to_time_counter, message_text)] = active[1]

        for message_text, (message, message_id, first_time_counter) in active_messages.items():
            if message_text not in to_active_messages:
                message_lifetime_by_id[message_id] = (message, first_time_counter, time_counter)

        configuration = to_configuration
        time_counter = to_time_counter
        active_messages = to_active_messages

    for message_text, (message, message_id, first_time_counter) in active_messages.items():
        if message_text not in to_active_messages:
            message_lifetime_by_id[message_id] = (message, first_time_counter, time_counter)

    return message_id_by_times, message_lifetime_by_id


def print_message_time_nodes(file: 'TextIO', message_id: int, message: Message, first_time: int, last_time: int) -> None:
    '''
    Print all the time nodes for a message exchanged between agents.
    '''
    prev_node = ''
    for time in range(first_time, last_time + 1):
        node = 'message-%s-%s' % (message_id, time)
        if prev_node != '':
            file.write('"%s" [ shape=box, penwidth=2, width=0, height=0, color=mediumblue ];\n' % node)
            file.write('"%s" -> "%s" [ penwidth=3, color=mediumblue, weight=1000, dir=forward, arrowhead=none ];\n'
                       % (prev_node, node))
        else:
            file.write('"%s" [ label="%s", shape=box, style=filled, color=paleturquoise ];\n' % (node, message.state))

    head_node = ''
    for time in range(0, first_time + 1):
        node = 'message-%s-%s' % (message_id, time)
        if time != first_time:
            file.write('"%s" [ shape=none, label="" ];\n' % node)
        if head_node != '':
            file.write('"%s" -> "%s" [ style=invis ];\n' % (head_node, node))
        head_node = node


def print_invalid_time_node(file: 'TextIO', invalid: Invalid) -> str:
    '''
    Print a node for a final invalid state message.
    '''
    node = str(invalid)
    file.write('"%s" [ label="%s", shape=box, style=filled, color=lightcoral ];\n'
               % (node, invalid_label(invalid)))
    return node


def print_agent_time_nodes(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-branches
    file: 'TextIO',
    transitions: List[Transition],
    configuration_by_name: Dict[str, Configuration],
    message_id_by_times: Dict[Tuple[int, str], int],
    agent_name: str,
    agent_index: int
) -> Tuple[str, Optional[str], Optional[str]]:
    '''
    Print the interaction nodes for a specific agent.
    '''

    file.write('subgraph "cluster_agent_%s" {\n' % agent_name)
    file.write('color = white;\n')
    file.write('fontsize = 24;\n')
    file.write('label = "%s";\n' % agent_name)

    time_counter = 0

    configuration = configuration_by_name[transitions[0].from_configuration_name]
    agent = configuration.agents[agent_index]
    is_deferring = agent.is_deferring()

    if is_deferring:
        color = 'indigo'
        penwidth = 6
    else:
        color = 'darkgreen'
        penwidth = 3
    color_is_dark = True

    mid_node = print_agent_state_node(file, time_counter, agent, color)

    last_message_node: Optional[str] = None
    last_message_name: Optional[str] = None

    time_counter += 1
    last_agent_node = print_agent_state_node(file, time_counter, agent, color, new_state=True)
    file.write('"%s" -> "%s" [ penwidth=%s, color=%s, weight=1000, dir=forward, arrowhead=none ];\n'
               % (mid_node, last_agent_node, penwidth, color))

    for transition in transitions:
        mid_time_counter = time_counter + 1
        to_time_counter = time_counter + 2
        to_configuration = configuration_by_name[transition.to_configuration_name]
        to_agent = to_configuration.agents[agent_index]
        to_deferring = to_agent.is_deferring()

        mid_node = '%s@%s' % (agent_name, mid_time_counter)

        did_message = False
        if len(to_configuration.messages_in_flight) > len(configuration.messages_in_flight):
            assert len(to_configuration.messages_in_flight) == len(configuration.messages_in_flight) + 1
            for message in new_messages(configuration, to_configuration):
                if message.source_agent_name == agent_name:
                    message_id = message_id_by_times[(to_time_counter, str(message))]
                    last_message_name = message.state.name
                    last_message_node = 'message-%s-%s' % (message_id, to_time_counter)
                    file.write('"%s":c -> "%s":c [ penwidth=3, color=mediumblue, constraint=false ];\n'
                               % (mid_node, last_message_node))
                    did_message = True

        message = transition.delivered_message
        if message.target_agent_name == agent_name:
            if message.state.name == 'time':
                message_node = print_time_message_node(file, message, time_counter)
            else:
                message_id = message_id_by_times[(time_counter, str(message))]
                message_node = 'message-%s-%s' % (message_id, time_counter)
            if did_message:
                arrowhead = 'none'
            else:
                arrowhead = 'normal'
            file.write('"%s":c -> "%s":c [ penwidth=3, color=mediumblue, dir=forward, arrowhead=%s ];\n'
                       % (message_node, mid_node, arrowhead))
            did_message = True

        print_agent_state_node(file, mid_time_counter, to_agent, color, did_message=did_message)

        new_state = agent.state != to_agent.state
        agent_node = print_agent_state_node(file, to_time_counter, to_agent, color, new_state=new_state)
        file.write('"%s" -> "%s" [ penwidth=%s, color=%s, weight=1000, dir=forward, arrowhead=none ];\n'
                   % (last_agent_node, mid_node, penwidth, color))
        if agent.state != to_agent.state:
            color_is_dark = not color_is_dark
            if to_deferring:
                penwidth = 6
                if color_is_dark:
                    color = 'indigo'
                else:
                    color = 'purple'
            else:
                penwidth = 3
                if color_is_dark:
                    color = 'darkgreen'
                else:
                    color = 'yellowgreen'

        file.write('"%s" -> "%s" [ penwidth=%s, color=%s, weight=1000, dir=forward, arrowhead=none ];\n'
                   % (mid_node, agent_node, penwidth, color))

        last_agent_node = agent_node

        time_counter = to_time_counter
        configuration = to_configuration
        agent = to_agent
        is_deferring = to_deferring

    mid_time_counter = time_counter + 1
    mid_node = print_agent_state_node(file, mid_time_counter, agent, color)

    file.write('"%s" -> "%s" [ penwidth=%s, color=%s, weight=1000, dir=forward, arrowhead=none ];\n'
               % (last_agent_node, mid_node, penwidth, color))
    last_agent_node = mid_node

    return last_agent_node, last_message_name, last_message_node


def print_time_message_node(file: 'TextIO', message: Message, time_counter: int) -> str:
    '''
    Print a node for a time message that triggered a transition.
    '''
    node = '%s-time-%s' % (message.target_agent_name, time_counter)
    file.write('"%s" [ label="time", shape=box, style=filled, color=paleturquoise ];\n' % node)
    return node


def print_agent_state_node(
    file: 'TextIO',
    time_counter: int,
    agent: Agent,
    color: str,
    *,
    new_state: bool = False,
    did_message: bool = False
) -> str:
    '''
    Print a node along an agent's timeline.
    '''
    node = '%s@%s' % (agent.name, time_counter)
    if new_state:
        file.write('"%s" [ shape=box, label="%s", style=filled, color=palegreen ];\n' % (node, agent.state))
    else:
        if did_message:
            penwidth = 4
            color = '"#0063cd"'
        else:
            penwidth = 2
        file.write('"%s" [ shape=box, label="", penwidth=%s, width=0, height=0, color=%s ];\n' % (node, penwidth, color))
    return node


class Model:
    '''
    Model the whole system.
    '''

    def __init__(
        self,
        agents: Collection[Agent],
        validate: Optional[Validation] = None
    ) -> None:
        #: How to validate configurations.
        self.validate = validate

        initial_configuration = self.validated_configuration(Configuration(agents=agents))

        #: Quick mapping from agent name to its index in the agents tuple.
        self.agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(initial_configuration.agents)}

        #: All the transitions between configurations.
        self.transitions: List[Transition] = []

        #: All the known configurations, keyed by their name.
        self.configurations = {initial_configuration.name: initial_configuration}

        if not initial_configuration.valid:
            return

        #: The names of all the configurations we didn't fully model yet.
        self.pending_configuration_names = [initial_configuration.name]

        while len(self.pending_configuration_names) > 0:
            self.explore_configuration(self.pending_configuration_names.pop())

    def explore_configuration(self, configuration_name: str) -> None:
        '''
        Explore all the transitions from a configuration.
        '''
        configuration = self.configurations[configuration_name]
        assert configuration.valid

        for agent in configuration.agents:
            self.deliver_message(configuration, Message.time(agent))

        for message_index, message in enumerate(configuration.messages_in_flight):
            self.deliver_message(configuration, message, message_index)

    def deliver_message(
        self,
        configuration: Configuration,
        message: Message,
        message_index: Optional[int] = None
    ) -> None:
        '''
        Deliver the specified message to its target, creating new transitions
        and, if needed, new pending configurations.
        '''
        agent_index = self.agent_indices[message.target_agent_name]
        agent = configuration.agents[agent_index]
        is_deferring = agent.is_deferring()

        actions: Optional[Collection[Action]] = None

        handler = getattr(agent, '_%s_when_%s' % (message.state.name, agent.state.name), None)
        if handler is not None:
            actions = handler(message)
        elif is_deferring:
            actions = ()

        if actions is None:
            self.missing_handler(configuration, agent, message, message_index)
            return

        if len(actions) == 0 and not is_deferring:
            raise RuntimeError('agent: %s in non-deferring state: %s defers message: %s'
                               % (agent.name, agent.state.name, message.state.name))

        for action in actions:
            self.perform_action(configuration, agent, agent_index, action, message, message_index)

    def perform_action(  # pylint: disable=too-many-arguments
        self,
        configuration: Configuration,
        agent: Agent,
        agent_index: int,
        action: Action,
        message: Message,
        message_index: Optional[int]
    ) -> None:
        '''
        Perform one of the actions the agent might take as a response to the
        message.
        '''
        if action.next_state is None:
            new_agent = None
            invalids: List[Invalid] = []
        else:
            new_agent = agent.with_state(action.next_state)
            reasons = new_agent.state.validate()
            if len(reasons) == 0:
                reasons = new_agent.validate()
            invalids = [Invalid(kind='agent', name=agent.name, reason=reason) for reason in reasons]

        for sent_message in action.send_messages:
            reasons = sent_message.state.validate()
            if len(reasons) == 0:
                reasons = sent_message.validate()
            for reason in reasons:
                invalids.append(Invalid(kind='message', name=sent_message.state.name, reason=reason))

        self.new_transition(configuration, new_agent, agent_index, message, message_index, invalids, action.send_messages)

    def missing_handler(
        self,
        configuration: Configuration,
        agent: Agent,
        message: Message,
        message_index: Optional[int]
    ) -> None:
        '''
        Report a missing message handler for an agent at some state.
        '''
        invalid = Invalid(kind='agent', name=agent.name,
                           reason='missing handler for message: %s when in state: %s' % (message.state, agent.state))

        self.new_transition(configuration, None, None, message, message_index, [invalid])

    def new_transition(  # pylint: disable=too-many-arguments
        self,
        from_configuration: Configuration,
        agent: Optional[Agent],
        agent_index: Optional[int],
        message: Message,
        message_index: Optional[int],
        invalids: Collection[Invalid],
        send_messages: Collection[Message] = ()
    ) -> None:
        '''
        Create a new transition, and, if needed, a new pending configuration.
        '''
        if agent is None:
            new_agents = from_configuration.agents
        else:
            assert agent_index is not None
            new_agents = tuple_replace(from_configuration.agents, agent_index, agent)

        if message_index is None:
            new_messages_in_flight = from_configuration.messages_in_flight
        else:
            new_messages_in_flight = tuple_remove(from_configuration.messages_in_flight, message_index)

        if len(send_messages) > 0:
            new_messages_in_flight = tuple(sorted(new_messages_in_flight + tuple(send_messages)))

        to_configuration = self.validated_configuration(Configuration(agents=new_agents,
                                                                      messages_in_flight=new_messages_in_flight,
                                                                      invalids=tuple(sorted(invalids))))

        if from_configuration == to_configuration:
            return

        transition = Transition(from_configuration_name=from_configuration.name,
                                delivered_message=message,
                                to_configuration_name=to_configuration.name)
        self.transitions.append(transition)

        if to_configuration.name in self.configurations:
            return

        self.configurations[to_configuration.name] = to_configuration

        if to_configuration.valid:
            self.pending_configuration_names.append(to_configuration.name)

    def validated_configuration(self, configuration: Configuration) -> Configuration:
        '''
        Attach all relevant :py:const:`Invalid` indicators to a
        :py:const:`Configuration`.
        '''
        if self.validate is None:
            return configuration

        reasons = self.validate(configuration)
        if len(reasons) == 0:
            return configuration

        invalids = [Invalid(kind='configuration', reason=reason) for reason in reasons]

        return Configuration(agents=configuration.agents, messages_in_flight=configuration.messages_in_flight, invalids=invalids)


def tuple_replace(data: Tuple, index: int, value: Any) -> Tuple:
    '''
    Return a new tuple which replaces a value at an index to a new value.

    This should have been a Python builtin.
    '''
    assert 0 <= index < len(data)
    return data[0:index] + (value,) + data[index + 1:]


def tuple_remove(data: Tuple, index: int) -> Tuple:
    '''
    Return a new tuple which removes a value at an index.

    This should have been a Python builtin.
    '''
    assert 0 <= index < len(data)
    return data[0:index] + data[index + 1:]


def main(
    *,
    flags: Optional[Callable[[ArgumentParser], None]] = None,
    model: Callable[[Namespace], Collection[Agent]],
    validate: Optional[Validation] = None,
    description: str = 'Investigate the total state space of communicating finite state machines',
    epilog: str = ''
) -> None:
    '''
    A universal main function for invoking the functionality provided by this
    package.

    Run with ``-h`` or ``--help`` for a full list of the options.
    '''
    parser = ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-o', '--output', action='store', metavar='FILE', help='Write output to the specified file.')
    parser.add_argument('-f', '--focus', metavar='AGENT', action='append', default=[],
                        help='Focus only on the specified agent. Repeat for focusing on multiple agents.')
    parser.add_argument('-n', '--names', action='store_true',
                        help='Keep only names (without the internal data).')
    parser.add_argument('-a', '--agents', action='store_true',
                        help='Show only agents (without the in-flight messages).')
    if flags is not None:
        flags(parser)

    subparsers = parser.add_subparsers(title='command', metavar='')

    states_parser = subparsers.add_parser('states', help='Print a list of all possible system states.',
                                          epilog='''
        Generate a simple list of all states, one per line.
    ''')
    states_parser.set_defaults(function=states_command)

    transitions_parser = subparsers.add_parser('transitions',
                                               help='Print a tab-separated file of all transitions between system states.',
                                               epilog='''
        Generate a tab-separated file, with headers, containing transitions between
        system states.  The columns in the file are: from_configuration_name,
        delivered_message_source_agent_name, delivered_message_name,
        delivered_message_data, delivered_message_target_agent_name, and
        to_configuration_name.

        By default lists all transitions. If two or more `--configuration PATTERN`
        flags are specified, generate a list showing the shortest path between the
        matching configurations.  The first pattern must match only a single
        configuration, to identify a unique starting point for the path.
    ''')
    transitions_parser.set_defaults(function=transitions_command)
    transitions_parser.add_argument('-m', '--messages', action='store_true',
                                    help='Do not show messages in configurations, and add a column for sent messages.')
    transitions_parser.add_argument('-c', '--configuration', metavar='PATTERN', action='append', default=[],
                                    help='Generate only a path going through a configuration matching the regexp pattern.')

    space_parser = subparsers.add_parser('space', help='Print a graphviz dot file visualizing the states space.',
                                         epilog='''
        Generate a graphviz `dot` diagram visualizing the states space. By default
        generates a diagram containing "everything", which has the advantage of
        completeness but is unwieldy for even simple systems. Typically one uses a
        combination of flags to restrict the amount of information included in the
        diagram. Selecting the right combination depends on both the model and what you
        are trying to achieve. See the README file for some examples.
    ''')
    space_parser.add_argument('-l', '--label', metavar='STR', default='Total Space', help='Specify a label for the graph.')
    space_parser.add_argument('-c', '--cluster', metavar='AGENT', action='append', default=[],
                              help='Cluster nodes according to the states of the specified agent. Repeat for nesting clusters.')
    space_parser.add_argument('-m', '--messages', action='store_true', help='Create separate nodes for messages.')
    space_parser.add_argument('-M', '--merge', action='store_true',
                              help='Merge nodes that only differ by in-flight messages.')
    space_parser.set_defaults(function=space_command)

    time_parser = subparsers.add_parser('time', help='Print a graphviz dot file visualizing an interaction path.',
                                         epilog='''
        Generate a graphviz `dot` diagram visualizing the interactions between agents
        along a path between configurations. This requires specifying the
        `--configuration` flag at least twice to identify the path.
    ''')
    time_parser.add_argument('-l', '--label', metavar='STR', default='Total Space', help='Specify a label for the graph.')
    time_parser.add_argument('-c', '--configuration', metavar='PATTERN', action='append', default=[],
                             help='Generate a path going through a configuration matching the regexp pattern.')
    time_parser.set_defaults(function=time_command)

    args = parser.parse_args(sys.argv[1:])
    system = System.compute(agents=model(args), validate=validate)
    if len(args.focus) > 0:
        system = system.focus_on_agents(args.focus)
    if args.names:
        system = system.only_names()
    if args.agents:
        system = system.only_agents()
    with output(args) as file:
        args.function(args, file, system)


def states_command(_args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``states`` command.
    '''
    system.print_states(file)


def transitions_command(args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``transitions`` command.
    '''
    if len(args.configuration) == 1:
        raise ValueError('configurations path must contain at least two patterns')
    system.print_transitions(file, args.configuration, args.messages)


def space_command(args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``space`` command.
    '''
    system.print_space(file, cluster_by_agents=args.cluster, label=args.label,
                       separate_messages=args.messages, merge_messages=args.merge)


def time_command(args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``time`` command.
    '''
    if len(args.configuration) < 2:
        raise ValueError('configurations path must contain at least two patterns')
    system.print_time(file, args.label, args.configuration)


@contextmanager
def output(args: Namespace) -> 'Iterator[TextIO]':
    '''
    Direct the output according to the ``--output`` command line flag.
    '''
    if args.output is None:
        yield sys.stdout
    else:
        with open(args.output, 'w') as file:
            yield file
