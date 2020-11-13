'''
A simple model of server with two clients.
'''

# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=wildcard-import
# pylint: disable=len-as-condition
# pylint: disable=no-else-return
# pylint: disable=unused-wildcard-import


from typing import Tuple
from total_space import *


# Trivial states for the client agent.
CLIENT_IDLE_STATE = State(name='idle', data=None)
CLIENT_WAIT_STATE = State(name='wait', data=None)


class ClientAgent(Agent):
    '''
    A simple client that talks with a singleton server.
    '''

    @staticmethod
    def idle(name: str) -> 'ClientAgent':
        '''
        Create a client in the initial (idle) state.
        '''
        return ClientAgent(name=name, state=CLIENT_IDLE_STATE)

    def response_actions(self, message: Message) -> Tuple[Action, ...]:
        if (self.state.name, message.name) == ('wait', 'time'):
            return ()

        if (self.state.name, message.name) == ('idle', 'time'):
            message = Message(source_agent_name=self.name, name='request', target_agent_name='server', data=self.name)
            return (
                Action(name='send_request', next_state=CLIENT_WAIT_STATE, send_messages=(message,)),
            )

        if (self.state.name, message.name) == ('wait', 'response'):
            assert message.data == self.name
            return (
                Action(name='receive_response', next_state=CLIENT_IDLE_STATE, send_messages=()),
            )

        raise self.unknown_message(message)


# Trivial states for the server agent.
#
# In general the server state data is a tuple of the names of the clients it got requests from.
#
# We therefore could have had just a single state name.
#
# Instead this uses a different name `ready` when the list is empty and `busy` when actually working
# on behalf of the 1st client in the tuple.
#
# In general it is always possible to have a single state name and just place all the information
# in the data field.
#
# However using clear state names makes it easier to understand the state machine logic.
SERVER_READY_STATE = State(name='ready', data=())


class ServerAgent(Agent):
    '''
    A simple server that handles multiple clients, one at a time.
    '''

    @staticmethod
    def ready(name: str) -> 'ServerAgent':
        '''
        Create a server in the initial (ready) state.
        '''
        return ServerAgent(name=name, state=SERVER_READY_STATE)

    def response_actions(self, message: Message) -> Tuple[Action, ...]:
# NOTE: Demonstrates missing logic (a partial model); uncomment for the full model.
#       if (self.state.name, message.name) == ('ready', 'time'):
#           return ()

        if message.name == 'request':  # NOTE: this is independent of the state name.
            return self.receive_request_actions(message)

        if (self.state.name, message.name) == ('busy', 'time'):
            return self.busy_time_actions()

        raise self.unknown_message(message)

    def receive_request_actions(self, message: Message) -> Tuple[Action, ...]:
        next_state = State(name='busy', data=self.state.data + (message.data,))
        return (
            Action(name='receive_request', next_state=next_state, send_messages=()),
        )

    def busy_time_actions(self) -> Tuple[Action, ...]:
        assert len(self.state.data) > 0
        done_client_name = self.state.data[0]
        remaining_client_names = self.state.data[1:]
        message = Message(source_agent_name=self.name,
                          name='response',
                          target_agent_name=done_client_name,
                          data=done_client_name)

        if len(remaining_client_names) == 0:
            return (
                Action(name='done', next_state=SERVER_READY_STATE, send_messages=(message,)),
            )
        else:
            next_state = State(name='busy', data=remaining_client_names)
            return (
                Action(name='next', next_state=next_state, send_messages=(message,)),
            )


# Investigate a system with a single server and two clients.
print_yaml(configuration_space([
    ClientAgent.idle('client-1'),
    ClientAgent.idle('client-2'),
    ServerAgent.ready('server')
]))
