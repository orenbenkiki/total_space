'''
Investigate the total state space of communicating state machines.
'''

# pylint: disable=C0330
# pylint: disable=inherit-non-class
# pylint: disable=len-as-condition
# pylint: disable=line-too-long
# pylint: disable=multiple-statements
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=pointless-statement
# pylint: disable=too-few-public-methods
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import


from argparse import ArgumentParser
from argparse import Namespace
import sys
from contextlib import contextmanager
from functools import total_ordering
from typing import *


__version__ = '0.1.2'


__all__ = [
    'State',
    'Message',
    'Action',
    'Agent',
    'Configuration',
    'Transition',
    'Model',
    'main',
]


@total_ordering
class Immutable:
    '''
    Prevent any properties from being modified.

    The code here is in a "functional" style and makes heavy use of immutable data.
    '''

    #: Allow modification of properties when initializing the object.
    is_initializing = False

    def __setattr__(self, name: str, value: Any) -> None:
        if Immutable.is_initializing:
            object.__setattr__(self, name, value)
        else:
            raise RuntimeError('trying to modify the property: %s of an immutable: %s'
                               % (name, self.__class__.__qualname__))

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def __lt__(self, other) -> bool:
        return str(self) < str(other)


@contextmanager
def initializing() -> Iterator[None]:
    '''
    Allow mutating :py:const:`Immutable` data during initialization.
    '''
    try:
        was_initalizing = Immutable.is_initializing
        Immutable.is_initializing = True
        yield
    finally:
        Immutable.is_initializing = was_initalizing


class State(Immutable):
    '''
    A state of an :py:const:`Agent`.

    In general the state of an agent could be "anything" (including "nothing" - None).
    It "should" be immutable; use named tuples for structures and simple tuples for arrays/lists.

    In general one can bury all the information in the data field and use a single state name.
    However for clarity, the overall space of all possible states is split into named sub-spaces.
    '''

    __slots__ = ['name', 'data']

    def __init__(self, *, name: str, data: Any = None) -> None:
        with initializing():
            #: The state's name, used for visualization and method names.
            self.name = name

            #: The state's actual data.
            self.data = data

    def validate(self) -> 'Collection[str]':  # pylint: disable=no-self-use
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


class Message(State):
    '''
    A message sent from one :py:const:`Agent` to another (or a time message).

    This has the same name-vs-data considerations as the agent :py:const:`State` above.
    '''

    __slots__ = ['source_agent_name', 'target_agent_name']

    def __init__(self, *, source_agent_name: str, name: str, target_agent_name: str, data: Any) -> None:
        State.__init__(self, name=name, data=data)
        with initializing():
            #: The name of the agent that generated the message, or ``time``.
            self.source_agent_name = source_agent_name

            #: The name of the agent that will receive the message.
            self.target_agent_name = target_agent_name

    @staticmethod
    def time(agent: 'Agent') -> 'Message':
        '''
        We shoehorn the time-passes event to look like a message whose source is the agent state name,
        whose name is ``time``, and whose data is empty (``None``).
        '''
        return Message(source_agent_name=agent.state.name, name='time', target_agent_name=agent.name, data=None)

    def __str__(self) -> str:
        return '%s -> %s -> %s' % (self.source_agent_name, State.__str__(self), self.target_agent_name)

    def only_names(self) -> 'Message':
        '''
        Remove the data for a simplified view of the message.
        '''
        return Message(source_agent_name=self.source_agent_name,
                       target_agent_name=self.target_agent_name,
                       name=self.name,
                       data=None)


class Action(Immutable):
    '''
    A possible action taken by an py:const:`Agent` as a response to receiving a :py:const:`Message`.
    '''

    __slots__ = ['name', 'next_state', 'send_messages']

    def __init__(self, *, name: str, next_state: Optional[State] = None, send_messages: 'Collection[Message]' = ()) -> None:
        with initializing():
            #: The name of the action for visualization.
            self.name = name

            #: The next :py:const:`State` of the agent. If ``None`` the agent remains in the same state.
            self.next_state = next_state

            #: Any :py:const:`Message` sent by the agent.
            #: We assume the communication fabric may reorder messages.
            #: That is, there's no guarantee at what order the sent messages will be received by their targets.
            self.send_messages = tuple(send_messages)


class Agent(Immutable):
    '''
    An agent in the :py:const:`Configuration`.

    Each agent is a non-deterministic state machine.

    Sub-classes should implement methods with the name ``_<message_name>_when_<state_name>``,
    which are invoked when the has a :py:const:`State` with ``state_name``,
    and receives a py:const:`Message` with the name ``message_name``.

    Each method takes the data of the message,
    returns a tuple of alternative :py:const:`Action` possibilities.
    '''

    __slots__ = ['name', 'state']

    def __init__(self, *, name: str, state: State) -> None:
        with initializing():
            #: The name of the agent for visualization.
            self.name = name

            #: The state of the agent.
            self.state = state

    def with_state(self, state: State) -> 'Agent':
        '''
        Return a new agent with a modified state.
        '''
        return self.__class__(name=self.name, state=state)

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
        agents: 'Collection[Agent]',
        messages_in_flight: 'Collection[Message]' = (),
        invalids: 'Collection[Invalid]' = ()
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

    def focus_on_agents(self, agent_indices: 'Collection[int]') -> 'Configuration':
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
        Remove all the data, keeping the names, for a simplified view of the configuration.
        '''
        agents = tuple(agent.only_names() for agent in self.agents)
        messages_in_flight = tuple(message.only_names() for message in self.messages_in_flight)
        invalids = self.invalids
        return Configuration(agents=agents, messages_in_flight=messages_in_flight, invalids=invalids)

    def only_agents(self) -> 'Configuration':
        '''
        Remove all the in-flight messages, keeping the agents, for a simplified view of the configuration.
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
        return '%s -> %s -> %s' % (self.from_configuration_name, self.delivered_message, self.to_configuration_name)


#: The type of a function that validates a :py:const:`Configuration`,
#: returning a hopefully empty collection of :py:const:`Invalid` indications.
Validation = Callable[[Configuration], 'Collection[Invalid]']

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
        agents: 'Collection[Agent]',
        validate: Optional[Validation] = None
    ) -> 'System':
        '''
        Compute the total state space of a system given some agents in their initial state.
        '''
        model = Model(agents, validate)
        return System(configurations=tuple(sorted(model.configurations.values())),
                      transitions=tuple(sorted(model.transitions)))

    def focus_on_agents(self, keep_agents: 'Collection[str]') -> 'System':
        '''
        Return a simplified view of the system which focuses a subset of the agents.
        '''
        assert len(keep_agents) > 0
        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(self.configurations[0].agents)}
        keep_indices = sorted([agent_indices[agent_name] for agent_name in keep_agents])

        return self.simplify(lambda configuration: configuration.focus_on_agents(keep_indices),
                             lambda message: message)

    def only_names(self) -> 'System':
        '''
        Strip all the data, keeping only the names, for a simpler view of the system.
        '''
        return self.simplify(lambda configuration: configuration.only_names(),
                             lambda message: message.only_names())

    def only_agents(self) -> 'System':
        '''
        Strip all the in-flight messages, keeping only the agents, for a simpler view of the system.
        '''
        return self.simplify(lambda configuration: configuration.only_agents(),
                             lambda message: message)

    def simplify(self,
                 simplify_configuration: Callable[[Configuration], Configuration],
                 simplify_message: Callable[[Message], Message]) -> 'System':
        '''
        Return a simplified view of the system.
        '''
        new_configuration_by_name = {}  # type: Dict[str, Configuration]
        new_name_by_old_name = {}  # type: Dict[str, str]

        for configuration in self.configurations:
            new_configuration = simplify_configuration(configuration)
            new_configuration_by_name[new_configuration.name] = new_configuration
            new_name_by_old_name[configuration.name] = new_configuration.name

        new_transitions = {}  # type: Dict[str, Transition]
        reachable_configuration_names = set()  # type: Set[str]
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

    def print_transitions(self, file: 'TextIO') -> None:
        '''
        Print a list of all the transitions between system configurations to a tab-separated file.
        '''
        file.write('from_configuration_name\t')
        file.write('delivered_message_source_agent_name\t')
        file.write('delivered_message_name\t')
        file.write('delivered_message_data\t')
        file.write('delivered_message_target_agent_name\t')
        file.write('to_configuration_name\n')

        for transition in self.transitions:
            file.write('%s\t' % transition.from_configuration_name)
            file.write('%s\t' % transition.delivered_message.source_agent_name)
            file.write('%s\t' % transition.delivered_message.name)
            file.write('%s\t' % transition.delivered_message.data)
            file.write('%s\t' % transition.delivered_message.target_agent_name)
            file.write('%s\n' % transition.to_configuration_name)

    def print_dot(
        self,
        file: 'TextIO',
        *,
        cluster_by_agents: 'Collection[str]' = (),
        label: str = 'Total Space',
        separate_messages: bool = False,
        merge_messages: bool = False
    ) -> None:
        '''
        Print a ``dot`` file visualizing all the possible system configuration states and the transitions between them.
        '''
        file.write('digraph G {\n')
        file.write('fontname = "Sans-Serif";\n')
        file.write('fontsize = 32;\n')
        file.write('node [fontname = "Sans-Serif"];\n')
        file.write('edge [fontname = "Sans-Serif"];\n')
        file.write('label = "%s";\n' % label)

        self.print_dot_nodes(file, cluster_by_agents, merge_messages)
        self.print_dot_edges(file, separate_messages=separate_messages, merge_messages=merge_messages)

        file.write('}\n')

    def print_dot_nodes(self, file: 'TextIO', cluster_by_agents: 'Collection[str]', merge_messages) -> None:
        '''
        Print all the nodes of the ``dot`` file.
        '''
        node_names = set()  # type: Set[str]
        if len(cluster_by_agents) == 0:
            for configuration in self.configurations:
                print_dot_node(file, configuration, merge_messages, node_names)
            return

        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(self.configurations[0].agents)}
        cluster_by_indices = [agent_indices[agent_name] for agent_name in cluster_by_agents]
        paths = [['%s @ %s' % (configuration.agents[agent_index].name, configuration.agents[agent_index].state.name)
                  for agent_index in cluster_by_indices]
                 for configuration in self.configurations]

        current_path = []  # type: List[str]
        for path, configuration in sorted(zip(paths, self.configurations)):
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

            print_dot_node(file, configuration, merge_messages, node_names)

        while len(current_path) > 0:
            current_path.pop()
            file.write('}\n')

    def print_dot_edges(  # pylint: disable=too-many-locals,too-many-branches
        self,
        file: 'TextIO',
        *,
        separate_messages: bool,
        merge_messages: bool
    ) -> None:
        '''
        Print all the edges of the ``dot`` file.
        '''
        if separate_messages:
            configuration_by_name = {configuration.name: configuration for configuration in self.configurations}

        message_edges = set()  # type: Set[str]
        message_nodes = set()  # type: Set[str]
        intermediate_nodes = set()  # type: Set[str]

        for transition in self.transitions:
            sent_message = None  # type: Optional[Message]
            if separate_messages:
                from_configuration = configuration_by_name[transition.from_configuration_name]
                to_configuration = configuration_by_name[transition.to_configuration_name]

                if len(to_configuration.messages_in_flight) > len(from_configuration.messages_in_flight):
                    assert len(to_configuration.messages_in_flight) == len(from_configuration.messages_in_flight) + 1
                    for message in to_configuration.messages_in_flight:
                        if message not in from_configuration.messages_in_flight:
                            sent_message = message

                if merge_messages:
                    from_configuration = from_configuration.only_agents()
                    to_configuration = to_configuration.only_agents()

            if not separate_messages:
                file.write('"%s" -> "%s" [penwidth=3, label="%s"];\n'
                           % (transition.from_configuration_name,
                              transition.to_configuration_name,
                              message_dot_label(transition.delivered_message)))
                continue

            if from_configuration.name == to_configuration.name:
                continue

            intermediate = '%s -> * -> %s' % (from_configuration.name, to_configuration.name)
            if intermediate not in intermediate_nodes:
                file.write('"%s" [ shape=point, width=0, height=0 ];\n' % intermediate)
                intermediate_nodes.add(intermediate)

            print_dot_message(file, transition.delivered_message, message_nodes)

            assert from_configuration.valid
            if to_configuration.valid:
                color = 'darkgreen'
            else:
                color = 'red'

            edges = [
                '"%s" -> "%s" [penwidth=3, color=%s];\n' % (from_configuration.name, intermediate, color),
                '"%s" -> "%s" [penwidth=3, color=blue];\n' % (message_dot_label(transition.delivered_message), intermediate),
                '"%s" -> "%s" [penwidth=3, color=%s];\n' % (intermediate, to_configuration.name, color)
            ]
            if sent_message is not None:
                print_dot_message(file, sent_message, message_nodes)
                edges.append('"%s" -> "%s" [penwidth=3, color=blue];\n' % (intermediate, message_dot_label(sent_message)))

            for edge in edges:
                if edge not in message_edges:
                    file.write(edge)
                    message_edges.add(edge)


def print_dot_node(file: 'TextIO', configuration: Configuration, merge_messages: bool, node_names: Set[str]) -> None:
    '''
    Print a node for a system configuration state.
    '''
    if merge_messages:
        configuration = configuration.only_agents()
    if configuration.name in node_names:
        return
    node_names.add(configuration.name)

    if configuration.valid:
        color = 'aquamarine'
        label = configuration.name.replace(' , ', '\n').replace(' ; ', '\n')
    else:
        color = 'lightcoral'
        label = '\n'.join([str(invalid) for invalid in configuration.invalids]).replace('because: ', 'because:\n')
    file.write('"%s" [ label="%s", shape=box, style=filled, fillcolor=%s ];\n' % (configuration.name, label, color))


def print_dot_message(file: 'TextIO', message: Message, message_nodes: Set[str]) -> None:
    '''
    Print a node for an in-flight messages.
    '''
    label = message_dot_label(message)
    if label in message_nodes:
        return
    message_nodes.add(label)
    file.write('"%s" [ label="{%s}", shape=record, style=filled, fillcolor=Turquoise ];\n' % (label, label))


def message_dot_label(message: Message) -> str:
    '''
    The label to show for a message.
    '''
    return '%s &rarr; | %s | &rarr; %s' % (message.source_agent_name, State.__str__(message), message.target_agent_name)


class Model:
    '''
    Model the whole system.
    '''

    def __init__(
        self,
        agents: 'Collection[Agent]',
        validate: Optional[Validation] = None
    ) -> None:
        #: How to validate configurations.
        self.validate = validate

        initial_configuration = self.validated_configuration(Configuration(agents=agents))

        #: Quick mapping from agent name to its index in the agents tuple.
        self.agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(initial_configuration.agents)}

        #: All the transitions between configurations.
        self.transitions = []  # type: List[Transition]

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
        Deliver the specified message to its target, creating new transitions and, if needed, new pending configurations.
        '''
        agent_index = self.agent_indices[message.target_agent_name]
        agent = configuration.agents[agent_index]
        handler = getattr(agent, '_%s_when_%s' % (message.name, agent.state.name), None)

        if handler is None:
            self.missing_handler(configuration, agent, message, message_index)
        else:
            for action in handler(message.data):
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
        Perform one of the actions the agent might take as a response to the message.
        '''
        if action.next_state is None:
            new_agent = None
            invalids = []  # type: List[Invalid]
        else:
            new_agent = agent.with_state(action.next_state)
            invalids = [Invalid(kind='agent', name=agent.name, reason=reason)
                         for reason in action.next_state.validate()]

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
                           reason='missing handler for message: %s when in state: %s' % (message.name, agent.state.name))

        self.new_transition(configuration, None, None, message, message_index, [invalid])

    def new_transition(  # pylint: disable=too-many-arguments
        self,
        from_configuration: Configuration,
        agent: Optional[Agent],
        agent_index: Optional[int],
        message: Message,
        message_index: Optional[int],
        invalids: 'Collection[Invalid]',
        send_messages: 'Collection[Message]' = ()
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
        Attach all relevant :py:const:`Invalid` indicators to a :py:const:`Configuration`.
        '''
        if self.validate is None:
            return configuration

        invalids = self.validate(configuration)
        if len(invalids) == 0:
            return configuration

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
    model: Callable[[Namespace], 'Collection[Agent]'],
    validate: Optional[Validation] = None,
    description: str = 'Investigate the total state space of communicating finite state machines',
    epilog: str = ''
) -> None:
    '''
    A universal main function for invoking the functionality provided by this package.

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

    states_parser = subparsers.add_parser('states', help='Print a list of all possible system states.')
    states_parser.set_defaults(function=states_command)

    transitions_parser = subparsers.add_parser('transitions',
                                               help='Print a tab-separated file of all transitions between system states.')
    transitions_parser.set_defaults(function=transitions_command)

    dot_parser = subparsers.add_parser('dot', help='Print a graphviz dot file visualizing the dot between system states.')
    dot_parser.add_argument('-l', '--label', metavar='STR', default='Total Space', help='Specify a label for the graph.')
    dot_parser.add_argument('-c', '--cluster', metavar='AGENT', action='append', default=[],
                            help='Cluster nodes according to the states of the specified agent. Repeat for nesting clusters.')
    dot_parser.add_argument('-m', '--messages', action='store_true', help='Create separate nodes for messages.')
    dot_parser.add_argument('-M', '--merge', action='store_true',
                            help='Merge nodes that only differ by in-flight messages.')
    dot_parser.set_defaults(function=dot_command)

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


def transitions_command(_args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``transitions`` command.
    '''
    system.print_transitions(file)


def dot_command(args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``dot`` command.
    '''
    system.print_dot(file, cluster_by_agents=args.cluster, label=args.label,
                     separate_messages=args.messages, merge_messages=args.merge)


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
