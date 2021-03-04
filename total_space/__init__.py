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
from copy import deepcopy
from queue import Queue
import re
import sys
from contextlib import contextmanager
from functools import total_ordering
from typing import *

try:
    from immutabledict import immutabledict  # type: ignore # pylint: disable=import-error
except ModuleNotFoundError:
    immutabledict = dict  # pylint: disable=invalid-name


__version__ = '0.2.7'


__all__ = [
    'RESERVED_NAMES',
    'Immutable',
    'modifier',
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


#: Do not use these as a name.
RESERVED_NAMES = ('init', 'time', 'any', 'deferring', 'none')

T = TypeVar('T')  # pylint: disable=invalid-name

class Memoize:
    '''
    Memoize objects to reduce memory usage.
    '''
    by_name: Dict[str, Any] = {}

    @staticmethod
    def memoize(obj: T) -> T:
        '''
        Return the memoized version of the object.

        This relies in ``str`` containing a full description of the object.
        '''
        name = str(obj)
        memoized = Memoize.by_name.get(name, obj)
        if memoized is None:
            memoized = Memoize.by_name[name] = obj
        return memoized


@total_ordering
class Immutable:
    '''
    Prevent any properties from being modified.

    The code here is in a "functional" style and makes heavy use of immutable
    data.
    '''

    #: Allow modification of properties of this object, e.g. when initializing it object.
    _initializing: Optional[int] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if Immutable._initializing == id(self) or Immutable._initializing == -1:
            object.__setattr__(self, name, value)
        else:
            raise RuntimeError(f'trying to modify the property: {name}\n'
                               f'of an immutable: {self.__class__.__qualname__}')

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __lt__(self, other) -> bool:
        return str(self) < str(other)


@contextmanager
def initializing(this: Any) -> Iterator[None]:
    '''
    Allow mutating :py:const:`Immutable` data during initialization.
    '''
    try:
        old_initalizing = Immutable._initializing
        if this is None:
            Immutable._initializing = -1
        else:
            Immutable._initializing = id(this)
        yield
    finally:
        Immutable._initializing = old_initalizing


Self = TypeVar('Self', bound='Immutable')

# MYPY: def modifier(function: Callable[Concatenate[Self, P], None]) -> Callable[Concatenate[Self, P], Self]:
def modifier(function: Callable) -> Callable:
    '''
    Wrap a method that modifies an immutable object, converting it to a method
    that returns a new object instead.

    The wrapped method may freely change the normally immutable data members.
    '''
    # MYPY: def _create_modified(this: Self, *args: P.args, **kwargs: P.kwargs) -> Self:
    def _create_modified(this: Self, *args: Any, **kwargs: Any) -> Self:
        with initializing(None):
            that = deepcopy(this)
        with initializing(that):
            function(that, *args, **kwargs)
        return Memoize.memoize(that)
    return _create_modified


class State(Immutable):
    '''
    A state of an :py:const:`Agent`.

    In general the state of an agent could be "anything" (including "nothing" -
    None). It "should" be immutable; use named tuples for structures and
    simple tuples for arrays/lists.

    In general one can bury all the information in the data field and use a
    single state name. However for clarity, the overall space of all possible
    states is split into named sub-spaces.
    '''

    __slots__ = ['name', 'data']

    def __init__(self, *, name: str, data: Any = None) -> None:
        with initializing(self):
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
        return f'{self.name} / {self.data}'

    def only_names(self) -> 'State':
        '''
        Remove the data for a simplified view of the message.
        '''
        return State(name=self.name, data=None)

    @modifier
    def with_name(self, name: str) -> None:
        '''
        Return a new state with a modified name.
        '''
        self.name = name

    @modifier
    def with_data(self, data: Immutable) -> None:
        '''
        Return a new state with a modified data.
        '''
        self.data = data


class Message(Immutable):
    '''
    A message sent from one :py:const:`Agent` to another (or a time message).

    This has the same name-vs-data considerations as the agent
    :py:const:`State` above.
    '''

    __slots__ = ['source_agent_name', 'target_agent_name', 'state']

    def __init__(self, *, source_agent_name: str, target_agent_name: str, state: State) -> None:
        with initializing(self):
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
        return Message(source_agent_name='@ ' + agent.state.name, target_agent_name=agent.name, state=State(name='time'))

    def __str__(self) -> str:
        return f'{self.source_agent_name} -> {self.state} -> {self.target_agent_name}'

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

    @modifier
    def with_name(self, name: str) -> None:
        '''
        Return a new state with a modified state with the new name.
        '''
        self.state = self.state.with_name(name)

    def is_immediate(self) -> bool:
        '''
        Return whether this is an immediate message (the name ends with ``!``).
        '''
        return self.state.name[-1] == '!'

    def is_replacement(self) -> bool:
        '''
        Return whether this message may replace an in-flight message.
        '''
        return '=>' in self.state.name

    def is_ordered(self) -> bool:
        '''
        Return whether this message is ordered relative to others between the same two agents.
        '''
        return '@' in self.state.name

    def order(self) -> int:
        '''
        Return the order of this message relative to others between the same two agents.

        Must only be invoked on ordered messages.
        '''
        parts = self.state.name.split('@')
        assert len(parts) > 1
        return int(parts[-1])

    @modifier
    def reorder(self, delta: int) -> None:
        '''
        Change the order of the message.

        Must only be invoked on ordered messages.
        '''
        parts = self.state.name.split('@')
        assert len(parts) > 1
        new_order = int(parts[-1]) + delta
        assert new_order >= 0
        parts[-1] = str(new_order)
        self.state = self.state.with_name('@'.join(parts))

    def clean_name(self) -> str:
        '''
        Return the clean message name (w/o the ``!`` or ``@`` suffix, or the ``=>`` prefix).
        '''
        message_name = self.state.name
        if self.is_immediate():
            message_name = message_name[:-1]
        if self.is_ordered():
            message_name = message_name.split('@')[0]
        index = message_name.rfind('=>')
        if index >= 0:
            message_name = message_name[index + 2:]
        return message_name


class Action(Immutable):
    '''
    A possible action taken by an :py:const:`Agent` as a response to receiving
    a :py:const:`Message`.
    '''

    __slots__ = ['name', 'next_state', 'send_messages']

    #: An action that does not change the state or send any messages.
    NOP: 'Action'

    def __init__(self, *, name: str, next_state: Optional[State] = None, send_messages: Collection[Message] = ()) -> None:
        with initializing(self):
            #: The name of the action for visualization.
            self.name = name

            #: The next :py:const:`State` of the agent. If ``None`` the agent remains in the same state.
            self.next_state = next_state

            #: Any :py:const:`Message` sent by the agent.
            #: We assume the communication fabric may reorder messages.
            #: That is, there's no guarantee at what order the sent messages will be received by their targets.
            #:
            #: If the name of a message ends with ``!``, then the message is
            #: taken to be "immediate", that is, it will be the 1st message to
            #: be delivered. Otherwise the message is added to the in-flight
            #: messages and may be delivered at any order.
            #:
            #: If the name of a message ends with ``@``, then the message is
            #: taken to be "ordered", that is, will only be delivered after any
            #: other ordered message with the same source and target agents.
            #:
            #: If the message contains ``=>`` then it is expected to contain a
            #: regexp before the ``=>`` and the message name following it. If
            #: there is a single existing mesages matching the regexp it will be
            #: removed and replaced by the new message. The regexp should also
            #: match the empty string if the message may be added as usual
            #: without replacing any existing message.
            self.send_messages = tuple(send_messages)

Action.NOP = Action(name='nop')


class Agent(Immutable):
    '''
    An agent in the :py:const:`Configuration`.

    Each agent is a non-deterministic state machine.

    Sub-classes should implement handler methods. When a message needs
    to be delivered to an agent, the code searches for a handler method
    in the following order:

    * ``_<message_name>_when_<state_name>``
    * ``_<message_name>_when_deferring`` (if :py:func:`Agent.is_deferring`)
    * ``_<message_name>_when_any``
    * ``_any_when_<state_name>``
    * ``_any_when_deferring`` (if :py:func:`Agent.is_deferring`)
    * ``_any_when_any``

    If catch-all handler methods are implemented, it is recommended that their
    first statement will check that the state and/or the message are in a list
    of expected values, and otherwise return :py:attr:`Agent.UNEXPECTED`. This
    will ensure that adding new states and/or messages will not be
    unintentionally handled by the catch-all handler method.

    Handler methods should have a single parameter for the message, and return
    either `:py:attr:`Agent.UNEXPECTED` (``None``) or a ``Collection`` of
    possible :py:const:`Action`, any one of which may be taken when receiving
    the message while at the current state. Providing multiple alternatives
    allows modeling a non-deterministic agent (e.g., modeling either a hit or a
    miss in a cache).

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

    def __init__(self, *, name: str, state: State,
                 max_in_flight_messages: int = 1,
                 children: Optional[Dict[str, State]] = None) -> None:
        if state.name in RESERVED_NAMES:
            raise RuntimeError(f'setting the reserved-named state: {state}\n'
                               f'for the agent: {name}')
        with initializing(self):
            #: The name of the agent for visualization.
            self.name = name

            #: The state of the agent.
            self.state = state

            #: The maximal number of simultaneous in-flight messages that may
            #: be generated by this agent.
            self.max_in_flight_messages = max_in_flight_messages

            #: The state of child agents.
            #:
            #: A child agent is an agent whose name starts with the name of
            #: this agent, followed by a ``-``. The key to the dictionary is
            #: the full child agent name.
            self.children = immutabledict(children or {})

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

    @modifier
    def with_state(self, state: State) -> None:
        '''
        Return a new agent with a modified state.
        '''
        self.state = state

    @modifier
    def with_children(self, children: Optional[Dict[str, State]]) -> None:
        '''
        Return a new agent with a modified state.
        '''
        self.children = immutabledict(children or {})

    def validate(self) -> Collection[str]:  # pylint: disable=no-self-use
        '''
        Return a hopefully empty collection of reasons the agent is invalid.

        This is only invoked if the state is valid as of itself.
        '''
        return ()

    def __str__(self) -> str:
        name = f'{self.name} @ {self.state.name}'
        if self.state.data not in [None, (), [], '']:
            name += ' / ' + str(self.state.data)
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

        with initializing(self):
            #: The kind of invalid condition (``agent``, ``message``, or a whole system ``configuration``).
            self.kind = kind

            #: The name of whatever is invalid.
            self.name = name

            #: The reason for the invalid condition (short one line text).
            self.reason = reason

    def __str__(self) -> str:
        if self.name is None:
            return f'{self.kind} is invalid because: {self.reason}'
        return f'{self.kind}: {self.name} is invalid because: {self.reason}'


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
        with initializing(self):
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
        agents = tuple(self.agents[agent_index] for agent_index in agent_indices)
        agent_names = {agent.name for agent in agents}
        messages_in_flight = tuple(message for message in self.messages_in_flight
                                   if message.source_agent_name in agent_names
                                   or message.target_agent_name in agent_names)
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
        with initializing(self):
            #: The name of the configuration before the transition.
            self.from_configuration_name = from_configuration_name

            #: The message that was delivered to an agent to trigger the transition.
            self.delivered_message = delivered_message

            #: The name of the configuration after the transition.
            self.to_configuration_name = to_configuration_name

    def __str__(self) -> str:
        return f'{self.from_configuration_name} => {self.delivered_message} => {self.to_configuration_name}'


#: The type of a function that validates a :py:const:`Configuration`,
#: returning a hopefully empty collection of reasons it is invalid.
Validation = Callable[[Configuration], Collection[str]]

class TimeTracking:
    '''
    Track data to generate time graphs.
    '''
    def __init__(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        transitions: List[Transition],
        configuration_by_name: Dict[str, Configuration]
    ) -> None:
        #: Assign unique ID to each message.
        self.message_id_by_times: Dict[Tuple[int, str], Tuple[int, Message]] = {}

        #: The first and last time each message is used.
        self.message_lifetime_by_id: Dict[int, Tuple[int, int, Message]] = {}

        #: The previous message to use for invisible edges to force node ranks.
        self.prev_message_nodes: Dict[Tuple[str, str, int], str] = {}

        #: The message that was replaced by any message, if any.
        self.replaced_message_id: Dict[int, int] = {}

        #: The nodes we already connected with invisible edges.
        self.connected_message_nodes: Set[Tuple[str, str]] = set()

        active_messages: Dict[str, Tuple[Message, int, int]] = {}

        time_counter = 1
        configuration = configuration_by_name[transitions[0].from_configuration_name]

        for message in configuration.messages_in_flight:
            message_text = str(message)
            if message_text in active_messages.keys():
                raise NotImplementedError('multiple in-flight instances of the same message: ' + message_text)
            message_id = len(active_messages)
            self.message_id_by_times[(0, message_text)] = (message_id, message)
            active_messages[message_text] = (message, message_id, 0)

            key = sorted([message.source_agent_name, message.target_agent_name])
            self.prev_message_nodes[(key[0], key[1], 0)] = f'message-{message_id}-0'

        next_message_id = len(active_messages)

        for transition in transitions:
            from_configuration = configuration_by_name[transition.from_configuration_name]
            to_configuration = configuration_by_name[transition.to_configuration_name]

            mid_time_counter = time_counter + 1
            to_time_counter = time_counter + 2
            to_active_messages: Dict[str, Tuple[Message, int, int]] = {}

            for message in to_configuration.messages_in_flight:
                message_text = str(message)
                if message_text in to_active_messages:
                    raise NotImplementedError('multiple in-flight instances of the same message: ' + message_text)

                active = active_messages.get(message_text)

                if message.is_ordered():
                    from_count = len([1 for other_message in from_configuration.messages_in_flight
                                      if other_message.is_ordered()
                                      and other_message.source_agent_name == message.source_agent_name
                                      and other_message.target_agent_name == message.target_agent_name])
                    to_count = len([1 for other_message in to_configuration.messages_in_flight
                                    if other_message.is_ordered()
                                    and other_message.source_agent_name == message.source_agent_name
                                    and other_message.target_agent_name == message.target_agent_name])
                    if to_count < from_count:
                        assert to_count == from_count - 1
                        prev_message = message.reorder(1)
                        active = active_messages[str(prev_message)]

                if active is None:
                    message_id = next_message_id
                    next_message_id += 1
                    active = (message, message_id, to_time_counter)
                else:
                    message_id = active[1]
                    self.message_id_by_times[(mid_time_counter, message_text)] = (message_id, message)
                to_active_messages[message_text] = active
                self.message_id_by_times[(to_time_counter, message_text)] = (message_id, message)

                key = sorted([message.source_agent_name, message.target_agent_name])
                self.prev_message_nodes[(key[0], key[1], to_time_counter)] = f'message-{message_id}-{to_time_counter}'

                if not message.is_replacement():
                    continue
                replaced_name = message.state.name[:message.state.name.index('=>')]
                replaced_id: Optional[int] = None
                for old_message in from_configuration.messages_in_flight:
                    if old_message.source_agent_name == message.source_agent_name \
                            and old_message.target_agent_name == message.target_agent_name \
                            and old_message.clean_name() == replaced_name:
                        assert replaced_id is None
                        replaced_id = self.message_id_by_times[(time_counter, str(old_message))][0]
                if replaced_id is not None:
                    self.replaced_message_id[message_id] = replaced_id

            configuration = to_configuration
            time_counter = to_time_counter
            active_messages = to_active_messages

        for (time, _text), (message_id, message) in self.message_id_by_times.items():
            lifetime = self.message_lifetime_by_id.get(message_id)
            if lifetime is None:
                lifetime = (time, time, message)
            else:
                assert lifetime[1] == time - 1
                lifetime = (lifetime[0], time, lifetime[2])
            self.message_lifetime_by_id[message_id] = lifetime


class System(Immutable):
    '''
    The total state space of the whole system.
    '''

    __slots__ = ['configurations', 'transitions']

    def __init__(
        self,
        *,
        initial_configuration: Configuration,
        configurations: Tuple[Configuration, ...],
        transitions: Tuple[Transition, ...]
    ) -> None:
        with initializing(self):
            #: The initial configuration.
            self.initial_configuration = initial_configuration

            #: All the possible configurations the system could be at.
            self.configurations = configurations

            #: All the transitions between the configurations.
            self.transitions = transitions

    @staticmethod
    def compute(
        *,
        agents: Collection[Agent],
        validate: Optional[Validation] = None,
        allow_invalid: bool = False,
        debug: bool = False,
        patterns: Optional[List['re.Pattern']] = None,
    ) -> 'System':
        '''
        Compute the total state space of a system given some agents in their
        initial state.
        '''
        model = Model(agents, validate, allow_invalid=allow_invalid, debug=debug, patterns=patterns)
        return System(initial_configuration=model.initial_configuration,
                      configurations=tuple(sorted(model.configurations.values())),
                      transitions=tuple(sorted(model.transitions)))

    def verify_reachable(self) -> None:
        '''
        Verify the initial configuration is reachable from every other configuration,
        which implies every configuration is reachable from every other configuration.
        '''
        incoming_transitions: Dict[str, List[Transition]] = {}
        for transition in self.transitions:
            transitions_list = incoming_transitions.get(transition.to_configuration_name)
            if transitions_list is None:
                transitions_list = incoming_transitions[transition.to_configuration_name] = []
            transitions_list.append(transition)

        if self.initial_configuration.name not in incoming_transitions:
            raise RuntimeError(f'there is no way to return to the initial configuration: {self.initial_configuration.name}')

        reachable_configurations: Set[str] = set()
        pending_configurations = [self.initial_configuration.name]

        while len(pending_configurations) > 0:
            configuration_name = pending_configurations.pop()
            if configuration_name in reachable_configurations:
                continue
            reachable_configurations.add(configuration_name)
            for transition in incoming_transitions[configuration_name]:
                pending_configurations.append(transition.from_configuration_name)

        if len(reachable_configurations) == len(self.configurations):
            return

        unreachable_configurations: List[str] = []
        for configuration in self.configurations:
            if configuration.name not in reachable_configurations:
                unreachable_configurations.append(configuration.name)

        raise RuntimeError(f'the initial configuration: {self.initial_configuration}\n'
                           f"can't be reached from {len(unreachable_configurations)} configurations:\n- "
                           + '\n- '.join(sorted(unreachable_configurations)))

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

        new_configurations = tuple(new_configuration_by_name[name] for name in sorted(reachable_configuration_names))
        new_initial_configuration = \
                new_configuration_by_name[new_name_by_old_name[self.initial_configuration.name]]
        return System(initial_configuration=new_initial_configuration,
                      configurations=new_configurations,
                      transitions=tuple(sorted(new_transitions.values())))

    def print_agents(self, file: 'TextIO') -> None:
        '''
        Print a list of all the system configurations to a file.
        '''
        for agent in sorted(self.initial_configuration.agents):
            file.write(agent.name)
            file.write('\n')

    def print_states(self, file: 'TextIO') -> None:
        '''
        Print a list of all the system configurations to a file.
        '''
        for configuration in self.configurations:
            file.write(configuration.name)
            file.write('\n')

    def print_transitions(self, file: 'TextIO', patterns: List['re.Pattern'], sent_messages: bool) -> None:
        '''
        Print a list of all the transitions between system configurations to a
        tab-separated file.
        '''
        if len(patterns) > 0:
            transitions: Collection[Transition] = self.transitions_path(patterns)
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

            file.write(from_configuration_name)
            file.write('\t')
            file.write(transition.delivered_message.source_agent_name)
            file.write('\t')
            file.write(transition.delivered_message.state.name)
            file.write('\t')
            file.write(str(transition.delivered_message.state.data))
            file.write('\t')
            file.write(transition.delivered_message.target_agent_name)
            file.write('\t')

            if sent_messages:
                messages = new_messages(from_configuration, to_configuration)
                if len(messages) > 0:
                    file.write(' , '.join([str(message) for message in messages]))
                    file.write('\t')
                else:
                    file.write('None\t')

            file.write(to_configuration_name)
            file.write('\n')

    def transitions_path(self, patterns: List['re.Pattern']) -> List[Transition]:
        '''
        Return the path of transitions between configurations matching the
        patterns.
        '''
        assert len(patterns) > 1

        skip_transitions = patterns[0] != _INIT_PATTERN
        if not skip_transitions:
            patterns = patterns[1:]

        configuration_name = self.initial_configuration.name

        outgoing_transitions: Dict[str, List[Transition]] = {}
        for transition in self.transitions:
            transitions_list = outgoing_transitions.get(transition.from_configuration_name)
            if transitions_list is None:
                transitions_list = outgoing_transitions[transition.from_configuration_name] = []
            transitions_list.append(transition)

        transitions: List[Transition] = []
        skip_transitions_count = 0
        for pattern in patterns:
            self.shortest_path(configuration_name, pattern, outgoing_transitions, transitions)
            configuration_name = transitions[-1].to_configuration_name
            if skip_transitions:
                skip_transitions = False
                skip_transitions_count = len(transitions)

        assert len(transitions) > skip_transitions_count
        return transitions[skip_transitions_count:]

    def shortest_path(
        self,
        from_configuration_name: str,
        to_pattern: 're.Pattern',
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

                for transition in outgoing_transitions.get(configuration_name, []):
                    far_transitions = near_transitions + [transition]
                    if transition.to_configuration_name in to_configuration_names:
                        transitions.extend(far_transitions)
                        return

                    far_pending.append((transition.to_configuration_name, far_transitions))

            if len(near_pending) == 0:
                near_pending = far_pending
                far_pending = []

        raise RuntimeError(f'there is no path from the configuration: {from_configuration_name}\n'
                           f'to a configuration matching the pattern: {to_pattern}')

    def matching_configuration_names(self, pattern: 're.Pattern') -> List[str]:
        '''
        Return all the names of the configurations that match a pattern.
        '''
        if pattern == _INIT_PATTERN:
            return [self.initial_configuration.name]

        configuration_names = [configuration.name for configuration in self.configurations if pattern.search(configuration.name)]
        if len(configuration_names) == 0:
            raise ValueError(f'the regexp pattern: {pattern}\n'
                             f'does not match any configurations')

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
        file.write(f'label = "{label}";\n')

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
            configurations = tuple(configuration.only_agents() for configuration in self.configurations)
        else:
            configurations = self.configurations

        if len(cluster_by_agents) == 0:
            for configuration in configurations:
                if configuration.name in reachable_configuration_names:
                    print_space_node(file, configuration, node_names)
            return

        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(configurations[0].agents)}
        cluster_by_indices = [agent_indices[agent_name] for agent_name in cluster_by_agents]
        paths = [[f'{configuration.agents[agent_index].name} @ {configuration.agents[agent_index].state.name}'
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
                file.write(f'subgraph "cluster_{" , ".join(current_path)}" {{\n')
                file.write('fontsize = 24;\n')
                file.write(f'label = "{cluster}";\n')

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
                label = message_space_label(delivered_message, None).replace(' | ', '\\n')
                file.write(f'"{from_configuration.name}" -> "{to_configuration.name}" '
                           f'[ penwidth=3, label="{label}" ];\n')
                continue

            edges: List[str] = []

            transition_context = f'{from_configuration.name} => {to_configuration.name}'
            if sent_message is None:
                intermediate = transition_context
            else:
                intermediate = f'{from_configuration.name} => {delivered_message} => {to_configuration.name}'

            if sent_message is not None:
                print_space_message(file, sent_message, message_nodes, transition_context)
                label = message_space_label(sent_message, transition_context)
                edges.append(f'"{intermediate}" -> "{label}" [ penwidth=3, color=mediumblue ];\n')
                arrowhead = 'none'
            else:
                arrowhead = 'normal'

            if known_target or sent_message is not None:
                print_space_message(file, delivered_message, message_nodes, transition_context)
                label = message_space_label(delivered_message, transition_context)
                edges.append(f'"{label}" -> "{intermediate}" '
                             f'[ penwidth=3, color=mediumblue, dir=forward, arrowhead={arrowhead} ];\n')

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
                f'"{from_configuration.name}" -> "{intermediate}" '
                f'[ penwidth=3, color={color}, dir=forward, arrowhead=none ];\n',
                f'"{intermediate}" -> "{intermediate}-2" '
                f'[ penwidth=3, color={color}, dir=forward, arrowhead=none ];\n'
                f'"{intermediate}-2" -> "{to_configuration.name}" '
                f'[ penwidth=3, color={color} ];\n'
            ]

            if intermediate not in intermediate_nodes:
                file.write(f'"{intermediate}" [ shape=box, label="", penwidth=4, width=0, height=0, color="#0063cd" ];\n')
                file.write(f'"{intermediate}-2" [ shape=box, label="", penwidth=2, width=0, height=0, color="darkgreen" ];\n')
                intermediate_nodes.add(intermediate)

            for edge in edges:
                if edge not in message_edges:
                    file.write(edge)
                    message_edges.add(edge)

    def print_time(  # pylint: disable=too-many-locals,too-many-branches
        self,
        file: 'TextIO',
        label: str,
        patterns: List['re.Pattern']
    ) -> None:
        '''
        Print a ``dot`` file visualizing the interaction between agents along
        the specified path.
        '''
        transitions = self.transitions_path(patterns)

        configuration_by_name = {configuration.name: configuration for configuration in self.configurations}
        agent_indices = {agent.name: agent_index for agent_index, agent in enumerate(self.configurations[0].agents)}

        time_tracking = TimeTracking(transitions, configuration_by_name)

        file.write('digraph G {\n')
        file.write('fontname = "Sans-Serif";\n')
        file.write('fontsize = 32;\n')
        file.write('node [ fontname = "Sans-Serif" ];\n')
        file.write('edge [ fontname = "Sans-Serif" ];\n')
        file.write(f'label = "{label}";\n')
        file.write('ranksep = 0.05;\n')

        final_configuration = configuration_by_name[transitions[-1].to_configuration_name]
        for invalid in final_configuration.invalids:
            print_invalid_time_node(file, invalid)

        sorted_agents = list(sorted(self.configurations[0].agents))

        agents_run: List[int] = []
        run_agent_indices: List[range] = []
        agent_index = 0
        while agent_index < len(sorted_agents):
            agent = sorted_agents[agent_index]
            run_index = len(run_agent_indices)
            agents_run.append(run_index)
            stop_index = agent_index + 1
            while stop_index < len(sorted_agents) and sorted_agents[stop_index].name.startswith(agent.name):
                agents_run.append(run_index)
                stop_index += 1
            run_agent_indices.append(range(agent_index, stop_index))
            agent_index = stop_index

        for left_agent_index, left_agent_run in enumerate(agents_run):
            for right_agent_index, right_agent_run in enumerate(agents_run):
                if left_agent_run == right_agent_run or right_agent_index <= left_agent_index:
                    continue
                print_message_time_nodes_between(file, sorted_agents[left_agent_index],
                                                 sorted_agents[right_agent_index], time_tracking)

        last_messages: List[Tuple[str, str, Optional[str], Optional[str]]] = []
        for run_index, run_agents_range in enumerate(reversed(run_agent_indices)):
            file.write(f'subgraph "cluster_between_{run_index}" {{\n')
            file.write('color = white;\n')
            file.write('fontsize = 0;\n')
            file.write('label = "";\n')

            for left_agent_index in run_agents_range:
                for right_agent_index in run_agents_range:
                    if right_agent_index <= left_agent_index:
                        continue
                print_message_time_nodes_between(file, sorted_agents[left_agent_index],
                                                 sorted_agents[right_agent_index], time_tracking)

            for agent_index in run_agents_range:
                agent = sorted_agents[agent_index]
                last_agent_node, last_message_name, last_message_node = \
                    print_agent_time_nodes(file, transitions, configuration_by_name, time_tracking,
                                           agent.name, agent_indices[agent.name])
                last_messages.append((agent.name, last_agent_node, last_message_name, last_message_node))

            file.write('}\n')

        for invalid in final_configuration.invalids:
            for agent_name, last_agent_node, last_message_name, last_message_node in last_messages:
                if invalid.kind == 'agent' and invalid.name == agent_name:
                    file.write(f'"{last_agent_node}" -> "{invalid}" [ penwidth=3, color=crimson, weight=1000 ];\n')
                elif invalid.kind == 'message' and invalid.name == last_message_name:
                    file.write(f'"{last_message_node}" -> "{invalid}" [ penwidth=3, color=crimson, weight=1000 ];\n')
                else:
                    file.write(f'"{last_agent_node}" -> "{invalid}" [ style=invis ];\n')

        file.write('}\n')


def new_messages(from_configuration: Configuration, to_configuration: Configuration) -> List[Message]:
    '''
    Return all the messages that exist in one configuration but not the other.
    '''
    return [message
            for message in to_configuration.messages_in_flight
            if not message_is_in(message, from_configuration)]

def message_is_in(message: Message, configuration: Configuration) -> bool:
    '''
    Return whether a message from an old configuration exists in a new configuration.
    '''
    if message in configuration.messages_in_flight:
        return True
    if not message.is_ordered():
        return False
    prev_message = message.reorder(1)
    if prev_message in configuration.messages_in_flight:
        return True
    return False

def print_space_node(file: 'TextIO', configuration: Configuration, node_names: Set[str]) -> None:
    '''
    Print a node for a system configuration state.
    '''
    if configuration.name in node_names:
        return
    node_names.add(configuration.name)

    if configuration.valid:
        color = 'palegreen'
        label = configuration.name.replace(' , ', '\\n').replace(' ; ', '\\n')
    else:
        color = 'lightcoral'
        label = '\\n\\n'.join([invalid_label(invalid) for invalid in configuration.invalids])
    file.write(f'"{configuration.name}" [ label="{label}", shape=box, style=filled, color={color}];\n')


def print_space_message(file: 'TextIO', message: Message, message_nodes: Set[str], context: Optional[str]) -> None:
    '''
    Print a node for an in-flight messages.
    '''
    show_label = message_space_label(message, None)
    label = message_space_label(message, context)
    if label in message_nodes:
        return
    message_nodes.add(label)
    if message.is_immediate():
        color = 'darkturquoise'
    else:
        color = 'paleturquoise'
    file.write(f'"{label}" [ label="{{{show_label}}}", shape=record, style=filled, color={color} ];\n')


def message_space_label(message: Message, context: Optional[str]) -> str:
    '''
    The label to show for a message.
    '''
    label = f'{message.source_agent_name} &#8594; ' \
            f'| {str(message.state).replace("=>", "&#8658;")} ' \
            f'| &#8594; {message.target_agent_name}'
    if context is not None:
        label += f" | {context}"
    return label


def invalid_label(invalid: Invalid) -> str:
    '''
    The label to show for an invalid notification.
    '''
    label = str(invalid)
    label = label.replace(' because:', '\\nbecause:')
    label = label.replace(' for message:', '\\nfor message:')
    label = label.replace(' when in state:', '\\nwhen in state:')
    return label


def print_message_time_nodes_between(  # pylint: disable=too-many-locals
    file: 'TextIO',
    left_agent: Agent,
    right_agent: Agent,
    time_tracking: TimeTracking
) -> None:
    '''
    Print all time nodes for messages between two agents.
    '''
    did_message = False

    for message_id, (first_time, last_time, message) in time_tracking.message_lifetime_by_id.items():
        if (message.source_agent_name != left_agent.name
                or message.target_agent_name != right_agent.name) \
            and (message.source_agent_name != right_agent.name
                or message.target_agent_name != left_agent.name):
            continue

        if not did_message:
            did_message = True
            file.write(f'subgraph "cluster_between_{left_agent.name}_and_{right_agent.name}" {{\n')
            file.write('color = white;\n')
            file.write('fontsize = 0;\n')
            file.write('label = "";\n')

        replaced_id = time_tracking.replaced_message_id.get(message_id)
        if replaced_id is None:
            continue

        replaced_last_time = time_tracking.message_lifetime_by_id[replaced_id][1]
        intermediate_time = replaced_last_time + 1
        message_first_time = time_tracking.message_lifetime_by_id[message_id][0]
        assert message_first_time == intermediate_time + 1
        replaced_node = f'message-{replaced_id}-{replaced_last_time}'
        intermediate_node = f'message-{message_id}-{intermediate_time}'
        message_node = f'message-{message_id}-{message_first_time}'
        time_tracking.connected_message_nodes.add((replaced_node, intermediate_node))
        time_tracking.connected_message_nodes.add((intermediate_node, message_node))
        file.write(f'"{intermediate_node}" [ shape=box, label="", penwidth=2, width=0, height=0, color=mediumblue ];\n')
        file.write(f'"{replaced_node}" -> "{intermediate_node}" [ penwidth=3, dir=forward, arrowhead=none, color=mediumblue ];\n')
        file.write(f'"{intermediate_node}" -> "{message_node}" [ penwidth=3, color=mediumblue ];\n')
        key = sorted([message.source_agent_name, message.target_agent_name])
        time_tracking.prev_message_nodes[(key[0], key[1], replaced_last_time)] = replaced_node
        time_tracking.prev_message_nodes[(key[0], key[1], intermediate_time)] = intermediate_node
        time_tracking.prev_message_nodes[(key[0], key[1], message_first_time)] = message_node

    if not did_message:
        return

    for message_id, (first_time, last_time, message) in time_tracking.message_lifetime_by_id.items():
        if (message.source_agent_name != left_agent.name
                or message.target_agent_name != right_agent.name) \
            and (message.source_agent_name != right_agent.name
                or message.target_agent_name != left_agent.name):
            continue

        print_message_time_nodes(file, message_id, message, first_time, last_time, time_tracking)

    file.write('}\n')


def print_message_time_nodes(  # pylint: disable=too-many-arguments,too-many-branches
    file: 'TextIO',
    message_id: int,
    message: Message,
    first_time: int,
    last_time: int,
    time_tracking: TimeTracking
) -> None:
    '''
    Print all the time nodes for a message exchanged between agents.
    '''
    key = sorted([message.source_agent_name, message.target_agent_name])

    if not message.is_replacement():
        for time in range(0, first_time + 1):
            node: Optional[str]
            if time == first_time:
                node = f'message-{message_id}-{time}'
            else:
                node = time_tracking.prev_message_nodes.get((key[0], key[1], time))
                if node is None:
                    node = f'message-{message_id}-{time}'
                    time_tracking.prev_message_nodes[(key[0], key[1], time)] = node
                    file.write(f'"{node}" [ shape=point, label="", style=invis ];\n')
            if time == 0:
                continue
            prev_node = time_tracking.prev_message_nodes[(key[0], key[1], time - 1)]
            connection = (prev_node, node)
            if connection in time_tracking.connected_message_nodes:
                continue
            time_tracking.connected_message_nodes.add(connection)
            file.write(f'"{prev_node}" -> "{node}" [ style=invis ];\n')

    prev_node = ''
    for time in range(first_time, last_time + 1):
        node = f'message-{message_id}-{time}'
        time_tracking.prev_message_nodes[(key[0], key[1], time)] = node
        if prev_node != '':
            file.write(f'"{node}" [ shape=box, label="", penwidth=2, width=0, height=0, color=mediumblue ];\n')
            time_tracking.connected_message_nodes.add((prev_node, node))
            file.write(f'"{prev_node}" -> "{node}" '
                       '[ penwidth=3, color=mediumblue, weight=1000, dir=forward, arrowhead=none ];\n')
        else:
            if message.is_immediate():
                color = 'darkturquoise'
            else:
                color = 'paleturquoise'
            if message.is_replacement():
                message = message.with_name(message.state.name[message.state.name.index('=>')+2:])
            file.write(f'"{node}" [ label="{message.state}", shape=box, style=filled, color={color} ];\n')
        prev_node = node


def print_invalid_time_node(file: 'TextIO', invalid: Invalid) -> str:
    '''
    Print a node for a final invalid state message.
    '''
    node = str(invalid)
    file.write(f'"{node}" [ label="{invalid_label(invalid)}", shape=box, style=filled, color=lightcoral ];\n')
    return node


def print_agent_time_nodes(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements,too-many-branches
    file: 'TextIO',
    transitions: List[Transition],
    configuration_by_name: Dict[str, Configuration],
    time_tracking: TimeTracking,
    agent_name: str,
    agent_index: int
) -> Tuple[str, Optional[str], Optional[str]]:
    '''
    Print the interaction nodes for a specific agent.
    '''

    file.write(f'subgraph "cluster_agent_{agent_name}" {{\n')
    file.write('color = white;\n')
    file.write('fontsize = 24;\n')
    file.write(f'label = "{agent_name}";\n')

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

    mid_node = print_agent_state_node(file, time_counter, agent, color, penwidth)

    last_message_node: Optional[str] = None
    last_message_name: Optional[str] = None

    time_counter += 1
    last_agent_node = print_agent_state_node(file, time_counter, agent, color, penwidth, new_state=True)
    file.write(f'"{mid_node}" -> "{last_agent_node}" '
               f'[ penwidth={penwidth}, color={color}, weight=1000, dir=forward, arrowhead=none ];\n')

    for transition in transitions:
        mid_time_counter = time_counter + 1
        to_time_counter = time_counter + 2
        to_configuration = configuration_by_name[transition.to_configuration_name]
        to_agent = to_configuration.agents[agent_index]
        to_deferring = to_agent.is_deferring()

        mid_node = f'{agent_name}@{mid_time_counter}'

        did_message = False
        if to_configuration.messages_in_flight != configuration.messages_in_flight:
            for message in new_messages(configuration, to_configuration):
                if message.source_agent_name == agent_name:
                    message_id = time_tracking.message_id_by_times[(to_time_counter, str(message))][0]
                    last_message_name = message.state.name
                    last_message_node = f'message-{message_id}-{to_time_counter}'
                    file.write(f'"{mid_node}":c -> "{last_message_node}":c '
                               '[ penwidth=3, color=mediumblue, constraint=false ];\n')
                    did_message = True

        message = transition.delivered_message
        if message.target_agent_name == agent_name:
            if message.state.name == 'time':
                message_node = print_time_message_node(file, message, time_counter)
            else:
                message_id = time_tracking.message_id_by_times[(time_counter, str(message))][0]
                message_node = f'message-{message_id}-{time_counter}'
            if did_message:
                arrowhead = 'none'
            else:
                arrowhead = 'normal'
            file.write(f'"{message_node}":c -> "{mid_node}":c '
                       f'[ penwidth=3, color=mediumblue, dir=forward, arrowhead={arrowhead} ];\n')
            did_message = True

        print_agent_state_node(file, mid_time_counter, to_agent, color, penwidth, did_message=did_message)

        new_state = agent.state != to_agent.state
        agent_node = print_agent_state_node(file, to_time_counter, to_agent, color, penwidth, new_state=new_state)
        file.write(f'"{last_agent_node}" -> "{mid_node}" '
                   f'[ penwidth={penwidth}, color={color}, weight=1000, dir=forward, arrowhead=none ];\n')
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

        file.write(f'"{mid_node}" -> "{agent_node}" '
                   f'[ penwidth={penwidth}, color={color}, weight=1000, dir=forward, arrowhead=none ];\n')

        last_agent_node = agent_node

        time_counter = to_time_counter
        configuration = to_configuration
        agent = to_agent
        is_deferring = to_deferring

    mid_time_counter = time_counter + 1
    mid_node = print_agent_state_node(file, mid_time_counter, agent, color, penwidth)

    file.write(f'"{last_agent_node}" -> "{mid_node}" '
               f'[ penwidth={penwidth}, color={color}, weight=1000, dir=forward, arrowhead=none ];\n')
    last_agent_node = mid_node

    file.write('}\n')

    return last_agent_node, last_message_name, last_message_node


def print_time_message_node(file: 'TextIO', message: Message, time_counter: int) -> str:
    '''
    Print a node for a time message that triggered a transition.
    '''
    node = f'{message.target_agent_name}-time-{time_counter}'
    if message.is_immediate():
        color = 'darkturquoise'
    else:
        color = 'paleturquoise'
    file.write(f'"{node}" [ label="time", shape=box, style=filled, color={color} ];\n')
    return node


def print_agent_state_node(
    file: 'TextIO',
    time_counter: int,
    agent: Agent,
    color: str,
    penwidth: int,
    *,
    new_state: bool = False,
    did_message: bool = False
) -> str:
    '''
    Print a node along an agent's timeline.
    '''
    node = f'{agent.name}@{time_counter}'
    if new_state:
        file.write(f'"{node}" [ shape=box, label="{agent.state}", style=filled, color=palegreen ];\n')
    else:
        if did_message:
            penwidth += 1
            color = '"#0063cd"'
        else:
            penwidth -= 1
        file.write(f'"{node}" [ shape=box, label="", penwidth={penwidth}, width=0, height=0, color={color} ];\n')
    return node

_INIT_PATTERN = re.compile('INIT')

class Model:  # pylint: disable=too-many-instance-attributes
    '''
    Model the whole system.
    '''

    def __init__(
        self,
        agents: Collection[Agent],
        validate: Optional[Validation] = None,
        *,
        allow_invalid: bool = False,
        debug: bool = False,
        patterns: Optional[List['re.Pattern']] = None,
    ) -> None:
        #: How to validate configurations.
        self.validate = validate

        #: Whether to allow invalid configurations (but do not further explore them).
        self.allow_invalid = allow_invalid

        #: Whether to print every created configuration to stderr for debugging.
        self.debug = debug

        if patterns is not None and len(patterns) == 2 and patterns[0] == re.compile('INIT'):
            pattern: Optional['re.Pattern'] = patterns[1]
        else:
            pattern = None

        #: The pattern of the configuration we are looking for.
        self.pattern = pattern

        #: Whether to abort building the model since we found the configuration we are looking for.
        self.abort = False

        agents = tuple(sorted(agents))

        #: Quick mapping from agent name to its index in the agents tuple.
        self.agent_indices = {agent.name: agent_index
                              for agent_index, agent
                              in enumerate(agents)}

        children, parents = Model.family_of_agents(agents)

        #: The children of each agent, if any.
        self.agent_children = children

        #: The parents of each agent, if any.
        self.agent_parents = parents

        agents = tuple(self.agent_with_children(agent, agents) for agent in agents)

        #: The initial configuration.
        self.initial_configuration = self.validated_configuration(Configuration(agents=agents))

        #: The furthest configuration we have reached so far along the specified patterns path.
        self.reachable_configuration_name = self.initial_configuration.name

        #: All the transitions between configurations.
        self.transitions: List[Transition] = []

        #: All the known configurations, keyed by their name.
        self.configurations = {self.initial_configuration.name: self.initial_configuration}

        if not self.initial_configuration.valid:
            return

        #: The names of all the configurations we didn't fully model yet.
        self.pending_configuration_names: Queue[str] = Queue()
        self.pending_configuration_names.put(self.initial_configuration.name)

        while not self.abort and not self.pending_configuration_names.empty():
            self.explore_configuration(self.pending_configuration_names.get())

    @staticmethod
    def family_of_agents(
        agents: Collection[Agent]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        '''
        Collect the children of each agent.
        '''
        agent_children: Dict[str, List[str]] = {agent.name: [] for agent in agents}
        agent_parents: Dict[str, List[str]] = {agent.name: [] for agent in agents}

        for parent_agent in agents:
            prefix = parent_agent.name + '-'
            for child_agent in agents:
                if not child_agent.name.startswith(prefix):
                    continue
                agent_children[parent_agent.name].append(child_agent.name)
                agent_parents[child_agent.name].append(parent_agent.name)

        return agent_children, agent_parents

    def agent_with_children(self, agent: Agent, agents: Tuple[Agent, ...]) -> Agent:
        '''
        Return the agent modified to contain the state of its children.
        '''
        child_names = self.agent_children[agent.name]
        child_agents = [agents[self.agent_indices[child_name]] for child_name in child_names]
        children = {child_agent.name: child_agent.state for child_agent in child_agents}
        return agent.with_children(children)

    def explore_configuration(self, configuration_name: str) -> None:
        '''
        Explore all the transitions from a configuration.
        '''
        configuration = self.configurations[configuration_name]
        assert configuration.valid

        delivered_immediate_messages = False
        for message_index, message in enumerate(configuration.messages_in_flight):
            if message.is_immediate():
                self.deliver_message(configuration, message, message_index)
                delivered_immediate_messages = True

        if delivered_immediate_messages:
            return

        for agent in configuration.agents:
            self.deliver_message(configuration, Message.time(agent))

        for message_index, message in enumerate(configuration.messages_in_flight):
            if not message.is_ordered() or message.order() == 0:
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

        handler = getattr(agent, f'_{message.clean_name()}_when_{agent.state.name}', None)
        if handler is None and is_deferring:
            handler = getattr(agent, f'_{message.clean_name()}_when_deferring', None)
        if handler is None:
            handler = getattr(agent, f'_{message.clean_name()}_when_any', None)
        if handler is None:
            handler = getattr(agent, f'_any_when_{agent.state.name}', None)
        if handler is None and is_deferring:
            handler = getattr(agent, '_any_when_deferring', None)
        if handler is None:
            handler = getattr(agent, '_any_when_any', None)

        if handler is not None:
            actions = handler(message)
        elif is_deferring:
            actions = ()

        if actions is None:
            self.missing_handler(configuration, agent, message, message_index)
            return

        if len(actions) == 0:
            if not is_deferring:
                raise RuntimeError(f'an agent: {agent.name}\n'
                                   f'in the non-deferring state: {agent.state.name}\n'
                                   f'defers the message: {message.state.name}')
            if message.is_immediate():
                raise RuntimeError(f'an agent: {agent.name}\n'
                                   f'in the deferring state: {agent.state.name}\n'
                                   f'defers the immediate message: {message.state.name}')

        for action in actions:
            self.perform_action(configuration, agent, agent_index, action, message, message_index)

    def perform_action(  # pylint: disable=too-many-arguments,too-many-branches
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
            new_agent: Optional[Agent] = None
            invalids_data: List[Tuple[Invalid, str]] = []
        else:
            new_agent = agent.with_state(action.next_state)
            assert isinstance(new_agent, Agent)
            reasons = new_agent.state.validate()
            if len(reasons) == 0:
                reasons = new_agent.validate()
            invalids_data = [(Invalid(kind='agent', name=agent.name, reason=reason), '')
                             for reason in reasons]

        send_messages = []
        for sent_message in action.send_messages:
            if sent_message.clean_name() in RESERVED_NAMES:
                raise RuntimeError(f'the reserved-named message: {sent_message}\n'
                                   f'is sent from the agent: {agent}')
            if sent_message.source_agent_name != agent.name:
                raise RuntimeError(f'the message: {sent_message}\n'
                                   f'pretends to be from: {sent_message.source_agent_name}\n'
                                   f'but is from: {agent}')
            if sent_message.target_agent_name not in self.agent_indices:
                raise RuntimeError(f'the message: {sent_message}\n'
                                   f'is sent to the unknown: {sent_message.target_agent_name}\n'
                                   f'from: {agent}')

            if sent_message.state.name[-1] == '@':
                max_prev = -1
                for prev_message in configuration.messages_in_flight:
                    if prev_message.is_ordered() \
                            and prev_message.source_agent_name == sent_message.source_agent_name \
                            and prev_message.target_agent_name == sent_message.target_agent_name:
                        max_prev = max(prev_message.order(), max_prev)
                sent_message = sent_message.with_name(f'{sent_message.state.name}{max_prev + 1}')
            else:
                sent_message = Memoize.memoize(sent_message)

            reasons = sent_message.state.validate()
            if len(reasons) == 0:
                reasons = sent_message.validate()
            for reason in reasons:
                invalids_data.append((Invalid(kind='message', name=sent_message.state.name, reason=reason), ''))

            send_messages.append(sent_message)

        self.new_transition(configuration, new_agent, agent_index, message, message_index, invalids_data, send_messages)

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
        reason = f'missing handler for message: {message.state} when in state: {agent.state}'
        invalid = Invalid(kind='agent', name=agent.name, reason=reason)

        self.new_transition(configuration, None, None, message, message_index, [(invalid, '')])

    def new_transition(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        self,
        from_configuration: Configuration,
        agent: Optional[Agent],
        agent_index: Optional[int],
        message: Message,
        message_index: Optional[int],
        invalids_data: List[Tuple[Invalid, str]],
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

            for parent_name in self.agent_parents[agent.name]:
                parent_index = self.agent_indices[parent_name]
                parent_agent = new_agents[parent_index]
                parent_agent = self.agent_with_children(parent_agent, new_agents)
                new_agents = tuple_replace(new_agents, parent_index, parent_agent)

            new_agents = Memoize.memoize(new_agents)

        if message_index is None:
            new_messages_in_flight = from_configuration.messages_in_flight
        else:
            new_messages_in_flight = tuple_remove(from_configuration.messages_in_flight, message_index)
            if message.is_ordered():
                assert message.order() == 0
                new_messages_in_flight = tuple(other_message.reorder(-1)
                                               if other_message.is_ordered()
                                               and other_message.source_agent_name == message.source_agent_name
                                               and other_message.target_agent_name == message.target_agent_name
                                               else other_message
                                               for other_message
                                               in new_messages_in_flight)

        new_send_messages = []
        for send_message in send_messages:
            if send_message.is_replacement():
                new_messages_in_flight = replace_message(new_messages_in_flight, send_message)
            else:
                new_send_messages.append(send_message)

        new_messages_in_flight = tuple(sorted(new_messages_in_flight + tuple(new_send_messages)))
        new_messages_in_flight = Memoize.memoize(new_messages_in_flight)

        if agent is not None:
            counted_messages = [in_flight_message for in_flight_message in new_messages_in_flight
                                if in_flight_message.source_agent_name == agent.name]
            in_flight_count = len(counted_messages)

            if in_flight_count > agent.max_in_flight_messages:
                reason = f'sending {in_flight_count} which is more than ' \
                         f'the maximal allowed {agent.max_in_flight_messages} messages'
                suffix = '\nmessages:\n- ' \
                    + '\n- '.join([str(counted_message) for counted_message in counted_messages])
                invalids_data.append((Invalid(kind='agent', name=agent.name, reason=reason), suffix))

        for invalid, suffix in invalids_data:
            if not self.allow_invalid:
                raise RuntimeError(f'in configuration: {from_configuration}\n'
                                   f'when delivering: {message}\n'
                                   f'then {invalid}'
                                   f'{suffix}')

        new_invalids = tuple(sorted([invalid for invalid, _suffix in invalids_data]))
        new_invalids = Memoize.memoize(new_invalids)

        to_configuration = self.validated_configuration(Configuration(agents=new_agents,
                                                                      messages_in_flight=new_messages_in_flight,
                                                                      invalids=new_invalids))

        if from_configuration == to_configuration:
            return

        to_configuration = Memoize.memoize(to_configuration)

        transition = Transition(from_configuration_name=from_configuration.name,
                                delivered_message=message,
                                to_configuration_name=to_configuration.name)
        self.transitions.append(transition)

        if to_configuration.name in self.configurations:
            return

        self.configurations[to_configuration.name] = to_configuration

        if self.pattern is not None and self.pattern.search(to_configuration.name):
            self.abort = True

        if to_configuration.valid:
            self.pending_configuration_names.put(to_configuration.name)
            if self.debug:
                sys.stderr.write(f'{to_configuration.name}\n')

    def is_reachable(self, from_configuration_name: str, to_configuration_name: str) -> bool:
        '''
        Test whether we have transitions between two configurations.

        When implemented, this will replace the path searching in the
        ``System`` class.
        '''
        sys.stderr.write(f'from {from_configuration_name}\n')
        sys.stderr.write(f'to {to_configuration_name}\n')
        if from_configuration_name == self.initial_configuration.name:
            return True
        raise NotImplementedError('path that does not start in the INIT phase')

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

        return Configuration(agents=configuration.agents,
                             messages_in_flight=configuration.messages_in_flight,
                             invalids=invalids)


def replace_message(messages_in_flight: Collection[Message], message: Message) -> Tuple[Message, ...]:
    '''
    Replace an existing message with a new one if requested.
    '''
    original_message = message
    index = message.state.name.index('=>')
    pattern = re.compile(message.state.name[:index])
    message = message.with_name(message.state.name[index + 2:])

    new_messages_in_flight: List[Message] = []
    replaced_message: Optional[Message] = None
    for message_in_flight in messages_in_flight:
        if message_in_flight.source_agent_name != message.source_agent_name \
                or message_in_flight.target_agent_name != message.target_agent_name:
            new_messages_in_flight.append(message_in_flight)
            continue
        message_in_flight_name = message_in_flight.state.name
        if '=>' in message_in_flight_name:
            index = message_in_flight_name.index('=>')
            message_in_flight_name = message_in_flight_name[index + 2:]
        if not pattern.match(message_in_flight_name):
            new_messages_in_flight.append(message_in_flight)
            continue
        if replaced_message is None:
            replaced_message = message_in_flight
            message = message.with_name(replaced_message.clean_name() + '=>' + message.state.name)
            continue
        raise RuntimeError(f'the replacement message: {original_message}\n'
                           f'can replace either the in-flight message: {replaced_message}\n'
                           f'or the in-flight-message: {message_in_flight}')

    new_messages_in_flight.append(message)

    if replaced_message is None:
        if not pattern.match('none'):
            raise RuntimeError(f'the replacement message: {original_message}\n'
                               f'did not replace any message')

    elif replaced_message.is_ordered() and message.is_ordered():
        replaced_order = replaced_message.order()

        def reorder_message(in_flight_message: Message) -> Message:
            if in_flight_message.is_ordered \
                    and in_flight_message.source_agent_name == message.source_agent_name \
                    and in_flight_message.target_agent_name == message.target_agent_name \
                    and in_flight_message.order() > replaced_order:
                return in_flight_message.reorder(-1)
            return in_flight_message

        new_messages_in_flight = [reorder_message(in_flight_message)
                                  for in_flight_message in new_messages_in_flight]

    return tuple(new_messages_in_flight)


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
    parser.add_argument('-i', '--invalid', action='store_true',
                         help='Allow invalid conditions (but do not further explore them).')
    parser.add_argument('-r', '--reachable', action='store_true',
                         help='Verify the initial state is reachable from every other state.')
    parser.add_argument('-d', '--debug', action='store_true',
                         help='Print every configuration as it is created to stderr for debugging.')
    parser.add_argument('-f', '--focus', metavar='AGENT', action='append', default=[],
                        help='Focus only on the specified agent. Repeat for focusing on multiple agents.')
    parser.add_argument('-n', '--names', action='store_true',
                        help='Keep only names (without the internal data).')
    parser.add_argument('-a', '--agents', action='store_true',
                        help='Show only agents (without the in-flight messages).')
    if flags is not None:
        flags(parser)

    subparsers = parser.add_subparsers(title='command', metavar='')

    states_parser = subparsers.add_parser('agents', help='Print a list of all system agents.',
                                          epilog='''
        Generate a simple list of all agents, one per line.
    ''')
    states_parser.set_defaults(function=agents_command)

    states_parser = subparsers.add_parser('states', help='Print a list of all possible system states.',
                                          epilog='''
        Generate a simple list of all states, one per line.
    ''')
    states_parser.set_defaults(function=states_command)

    transitions_parser = subparsers.add_parser('transitions',
                                               help='Print a tab-separated file of all transitions between system states.',
                                               epilog='''
        Generate a tab-separated file, with headers, containing transitions between
        system states. The columns in the file are: from_configuration_name,
        delivered_message_source_agent_name, delivered_message_name,
        delivered_message_data, delivered_message_target_agent_name, and
        to_configuration_name.

        By default lists all transitions. If two or more `--configuration
        PATTERN` flags are specified, generate a list showing the shortest path
        between the matching configurations.
    ''')
    transitions_parser.set_defaults(function=transitions_command)
    transitions_parser.add_argument('-m', '--messages', action='store_true',
                                    help='Do not show messages in configurations, and add a column for sent messages.')
    transitions_parser.add_argument('-c', '--configuration', metavar='PATTERN', action='append', default=[],
                                    help='Generate only a path going through a configuration matching the regexp pattern. '
                                         'The special pattern INIT matches the initial configuration.')

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
    if 'function' not in args:
        raise RuntimeError('no command specified\n'
                           'run with --help for a list of commands')
    if 'configuration' in args:
        args.patterns = [re.compile(pattern) for pattern in args.configuration]
    else:
        args.patterns = None
    system = System.compute(agents=model(args), validate=validate, allow_invalid=args.invalid,
                            debug=args.debug, patterns=args.patterns)
    if args.reachable:
        system.verify_reachable()
    if len(args.focus) > 0:
        system = system.focus_on_agents(args.focus)
    if args.names:
        system = system.only_names()
    if args.agents:
        system = system.only_agents()
    with output(args) as file:
        args.function(args, file, system)


def agents_command(_args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``agents`` command.
    '''
    system.print_agents(file)


def states_command(_args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``states`` command.
    '''
    system.print_states(file)


def transitions_command(args: Namespace, file: 'TextIO', system: System) -> None:
    '''
    Implement the ``transitions`` command.
    '''
    if len(args.patterns) == 1:
        raise ValueError('configurations path must contain at least two patterns')
    system.print_transitions(file, args.patterns, args.messages)


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
    if len(args.patterns) < 2:
        raise ValueError('configurations path must contain at least two patterns')
    system.print_time(file, args.label, args.patterns)


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
