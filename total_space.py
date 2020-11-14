'''
Investigate the total state space of communicating state machines.
'''

# pylint: disable=missing-docstring
# pylint: disable=inherit-non-class
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=C0330
# pylint: disable=multiple-statements
# pylint: disable=len-as-condition
# pylint: disable=pointless-statement
# pylint: disable=no-member


from abc import abstractmethod
from warnings import warn
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple
from typing import Tuple


__all__ = [
    'State',
    'Message',
    'Action',
    'Agent',
    'Configuration',
    'Transition',
    'Node',
    'configuration_space',
    'print_yaml',
]


# The code here uses tuples instead of lists.
#
# The code here is in a "functional" style and makes heavy use of immutable data.
#
# It therefore uses named tuples instead of simple classes and tuples instead of lists.
#
# Using python 3.6 would have allowed using a much cleaner syntax for the named tuples.


# A state of an agent.
#
# In general the state of an agent could be "anything" (including "nothing" - None).
#
# It "should" be immutable; use named tuples for structures and simple tuples for arrays/lists.
#
# In general one can bury all the information in the data field and use a single state name.
#
# However for clarity, the overall space of all possible states is split into named sub-spaces.

State = NamedTuple('State', [
    ('name', 'str'),
    ('data', 'Any'),
])

# A message sent between two agents.
#
# This has the same name-vs-data considerations as the agent state above.
#
# We shoehorn the time-passes event to look like a message whose source is the agent state name,
# whose name is `time`, and whose data is empty (`None`).

MessageBase = NamedTuple('MessageBase', [
    ('source_agent_name', 'str'),
    ('name', 'str'),
    ('target_agent_name', 'str'),
    ('data', 'Any'),
])


class Message(MessageBase):
    def text(self) -> str:
        if self.data not in [None, (), [], '']:
            data = ' / %s' % (self.data)
        else:
            data = ''
        return '%s -> %s%s -> %s' % (self.source_agent_name, self.name, data, self.target_agent_name)

    @staticmethod
    def time(agent: 'Agent') -> 'Message':
        return Message(source_agent_name=agent.state.name, name='time', target_agent_name=agent.name, data=None)


# An action taken by an agent as a response to an event.
#
# The agent can change its internal state and/or send messages.
#
# We assume the communication fabric may reorder messages.
#
# That is, there's no guarantee at what order the sent messages will be received by their targets.

Action = NamedTuple('Action', [
    ('name', 'str'),
    ('next_state', 'State'),
    ('send_messages', 'Tuple[Message, ...]'),
])


# An agent in the system.
#
# Each agent is a state machine.

AgentBase = NamedTuple('AgentBase', [
    ('name', 'str'),
    ('state', 'State'),
])

class _MissingLogicError(NotImplementedError):
    pass

class Agent(AgentBase):
    def text(self) -> str:
        name = '%s @ %s' % (self.name, self.state.name)
        if self.state.data not in [None, (), [], '']:
            name += ' / %s' % (self.state.data,)
        return name

    @abstractmethod
    def response_actions(self, message: Message) -> Tuple[Action, ...]:
        '''
        Given the current state of the agent, and a message received by it, return the potential alternative actions that might be taken.

        That is, this is a non-deterministic state machine.

        This is especially important when dealing with time events.
        '''

    def unknown_message(self, message: Message) -> _MissingLogicError:
        '''
        Used to report missing logic (partial model) when receiving a message.
        '''
        return _MissingLogicError('unexpected message: %s in state: %s for agent: %s'
                                  % (message.name, self.state.name, self.name))

# A configuration (total state) of the system.
#
# This includes all the agents and all the in-flight messages.

ConfigurationBase = NamedTuple('ConfigurationBase', [
    ('name', 'str'),
    ('agents', 'Tuple[Agent, ...]'),
    ('messages_in_flight', 'Tuple[Message, ...]'),
    ('missing_message', 'Optional[Message]'),
])

class Configuration(ConfigurationBase):
    @staticmethod
    def new(agents: List[Agent]) -> 'Configuration':
        assert len(agents) > 0
        return Configuration(name='', agents=tuple(sorted(agents)), messages_in_flight=(), missing_message=None).rename()

    def rename(self) -> 'Configuration':
        name = ' , '.join([agent.text() for agent in self.agents])
        if len(self.messages_in_flight) > 0:
            name += ' ; '
            name += ' , '.join([message.text() for message in self.messages_in_flight])
        if self.missing_message is not None:
            name += ' ! ' + self.missing_message.text()
        return self._replace(name=name)


# A transition between configurations.

Transition = NamedTuple('Transition', [
    ('from_configuration_name', 'str'),
    ('received_message', 'Message'),
    ('to_configuration_name', 'str'),
])


# A node in the total state space graph.

Node = NamedTuple('Node', [
    ('configuration', 'Configuration'),
    ('outgoing_transitions', 'Tuple[Transition, ...]'),
    ('incoming_transitions', 'Tuple[Transition, ...]'),
])


def configuration_space(agents: List[Agent]) -> Dict[str, Node]:
    '''
    Return the total state space for a system containing the specified agents.

    The key for each node is the configuration name.
    '''
    initial_configuration = Configuration.new(agents)
    agent_index_by_name = {agent.name: agent_index for agent_index, agent in enumerate(initial_configuration.agents)}
    initial_node = Node(initial_configuration, (), ())
    nodes = {initial_node.configuration.name: initial_node}
    pending = [initial_configuration.name]
    while len(pending) > 0:
        node = nodes[pending.pop()]
        _explore_time(pending, nodes, node)
        _explore_messages(pending, nodes, node, agent_index_by_name)
    return nodes


def _explore_time(
    pending: List[str],
    nodes: Dict[str, Node],
    node: Node,
) -> None:
    for agent_index, agent in enumerate(node.configuration.agents):
        message = Message.time(agent)
        try:
            for action in agent.response_actions(message):
                _apply_action(pending, nodes, node, agent_index, agent, action, Message.time(agent))
        except _MissingLogicError as missing_logic:
            _record_missing_logic(nodes, node, message)
            warn(str(missing_logic))

def _explore_messages(
    pending: List[str],
    nodes: Dict[str, Node],
    node: Node,
    agent_index_by_name: Dict[str, int],
) -> None:
    for message_index, message in enumerate(node.configuration.messages_in_flight):
        agent_index = agent_index_by_name[message.target_agent_name]
        agent = node.configuration.agents[agent_index]
        try:
            for action in agent.response_actions(message):
                _apply_action(pending, nodes, node, agent_index, agent, action, message, message_index)
        except _MissingLogicError as missing_logic:
            _record_missing_logic(nodes, node, message)
            warn(str(missing_logic))


def _apply_action(  # pylint: disable=too-many-arguments,too-many-locals
    pending: List[str],
    nodes: Dict[str, Node],
    node: Node,
    agent_index: int,
    agent: Agent,
    action: Action,
    message: Message,
    message_index: Optional[int] = None,
) -> None:
    new_agent = agent._replace(state=action.next_state)
    new_agents = _replace(node.configuration.agents, agent_index, new_agent)

    new_messages_in_flight = node.configuration.messages_in_flight
    if message_index is not None:
        new_messages_in_flight = _remove(new_messages_in_flight, message_index)
    new_messages_in_flight = tuple(sorted(new_messages_in_flight + action.send_messages))

    new_configuration = node.configuration._replace(agents=new_agents, messages_in_flight=new_messages_in_flight).rename()

    transition = Transition(from_configuration_name=node.configuration.name,
                            received_message=message,
                            to_configuration_name=new_configuration.name)

    new_outgoing_transitions = tuple(sorted(node.outgoing_transitions)) + (transition,)
    node = node._replace(outgoing_transitions=new_outgoing_transitions)
    nodes[node.configuration.name] = node

    old_node = nodes.get(new_configuration.name)
    if old_node is None:
        new_node = Node(new_configuration, incoming_transitions=(transition,), outgoing_transitions=())
        pending.append(new_node.configuration.name)

    else:
        new_incoming_transitions = tuple(sorted(old_node.incoming_transitions)) + (transition,)
        new_node = old_node._replace(incoming_transitions=new_incoming_transitions)

    nodes[new_node.configuration.name] = new_node


def _record_missing_logic(
    nodes: Dict[str, Node],
    node: Node,
    message: Message,
) -> None:
    new_configuration = node.configuration._replace(missing_message=message).rename()

    transition = Transition(from_configuration_name=node.configuration.name,
                            received_message=message,
                            to_configuration_name=new_configuration.name)

    nodes[new_configuration.name] = Node(configuration=new_configuration,
                                         incoming_transitions=(transition,),
                                         outgoing_transitions=())


def _replace(data: Tuple, index: int, datum: Any) -> Tuple:
    return data[0:index] + (datum,) + data[index + 1:]


def _remove(data: Tuple, index: int) -> Tuple:
    return data[0:index] + data[index + 1:]


def print_yaml(nodes: Dict[str, Node]):
    node_index_by_name = {}
    node_name_by_index = []
    for index, name in enumerate(sorted(nodes.keys())):
        node_index_by_name[name] = index
        node_name_by_index.append(name)

    for node_index, node_name in enumerate(node_name_by_index):
        node = nodes[node_name]
        print('%s:' % node_index)
        print('  name: "%s"' % node.configuration.name)
        print('  incoming:')
        for transition in node.incoming_transitions:
            print('    %s:' % node_index_by_name[transition.from_configuration_name])
            print('      source: %s' % transition.received_message.source_agent_name)
            print('      message: %s' % transition.received_message.name)
            print('      target: %s' % transition.received_message.target_agent_name)
        print('  outgoing:')
        for transition in node.outgoing_transitions:
            print('    %s:' % node_index_by_name[transition.to_configuration_name])
            print('      source: %s' % transition.received_message.source_agent_name)
            print('      message: %s' % transition.received_message.name)
            print('      target: %s' % transition.received_message.target_agent_name)
