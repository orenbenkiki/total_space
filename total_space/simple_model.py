'''
A simple model of server with two clients.
'''

# pylint: disable=global-statement
# pylint: disable=invalid-name
# pylint: disable=len-as-condition
# pylint: disable=line-too-long
# pylint: disable=no-else-return
# pylint: disable=no-name-in-module
# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import


from argparse import ArgumentParser
from argparse import Namespace
from typing import *
from total_space import *


#: Trivial state for the client agent.
CLIENT_IDLE_STATE = State(name='idle', data=None)

#: Trivial state for the client agent.
CLIENT_WAIT_STATE = State(name='wait', data=None)


class ClientAgent(Agent):
    '''
    A simple client that talks with a singleton server.
    '''

    @staticmethod
    def new(name: str) -> 'ClientAgent':
        '''
        Create a client in the initial (idle) state.
        '''
        return ClientAgent(name=name, state=CLIENT_IDLE_STATE)

    def _time_when_wait(self, _data: Any) -> Tuple[Action, ...]:
        return ()

    def _time_when_idle(self, _data: Any) -> Tuple[Action, ...]:
        request = Message(source_agent_name=self.name, name='request', target_agent_name='server', data=self.name)
        return (
            Action(name='send_request', next_state=CLIENT_WAIT_STATE, send_messages=(request,)),
        )

    def _response_when_wait(self, data: Any) -> Tuple[Action, ...]:
        assert data == self.name
        return (
            Action(name='receive_response', next_state=CLIENT_IDLE_STATE, send_messages=()),
        )


class InvalidServerState(State):
    '''
    A server state that reports invalid errors, for tests.
    '''

    def validate(self) -> 'Collection[str]':
        if len(self.data) == 2:
            return ['hold two requests']
        return ()


#: Trivial state for the server agent.
#:
#: In general the server state data is a tuple of the names of the clients it got requests from.
#:
#: We therefore could have had just a single state name.
#:
#: Instead this uses a different name `ready` when the list is empty and `busy` when actually working
#: on behalf of the 1st client in the tuple.
#:
#: In general it is always possible to have a single state name and just place all the information
#: in the data field.
#:
#: However using clear state names makes it easier to understand the state machine logic.
SERVER_READY_STATE = State(name='ready', data=())


#: Class to use for server states.
SERVER_STATE = State  # type: Type[State]


class PartialServerAgent(Agent):
    '''
    A partial server (for testing).

    The server handles multiple clients, one at a time.
    '''

    @classmethod
    def new(cls, name: str) -> 'Agent':
        '''
        Create a server in the initial (ready) state.
        '''
        return cls(name=name, state=SERVER_READY_STATE)

    def _time_when_busy(self, _data: Any) -> Tuple[Action, ...]:
        assert len(self.state.data) > 0
        done_client_name = self.state.data[0]
        remaining_client_names = self.state.data[1:]
        response = Message(source_agent_name=self.name,
                           name='response',
                           target_agent_name=done_client_name,
                           data=done_client_name)

        if len(remaining_client_names) == 0:
            return (
                Action(name='done', next_state=SERVER_READY_STATE, send_messages=(response,)),
            )
        else:
            next_state = SERVER_STATE(name='busy', data=remaining_client_names)
            return (
                Action(name='next', next_state=next_state, send_messages=(response,)),
            )

    def _request_when_ready(self, data: Any) -> Tuple[Action, ...]:
        return self._request_when_any(data)

    def _request_when_busy(self, data: Any) -> Tuple[Action, ...]:
        return self._request_when_any(data)

    def _request_when_any(self, data: Any) -> Tuple[Action, ...]:
        next_state = SERVER_STATE(name='busy', data=self.state.data + (data,))
        return (
            Action(name='receive_request', next_state=next_state, send_messages=()),
        )


class FullServerAgent(PartialServerAgent):
    '''
    A full server.
    '''

    def _time_when_ready(self, _data: Any) -> Tuple[Action, ...]:
        return ()


def flags(parser: ArgumentParser) -> None:
    '''
    Add command line flags for testing partial and invalid models.
    '''
    group = parser.add_argument_group('model')
    group.add_argument('-p', '--partial', action='store_true', help='Generate a partial model.')
    group.add_argument('-i', '--invalid', action='store_true', help='Generate an invalid model.')
    group.add_argument('-c', '--clients', metavar='NUMBER', type=int, default=2, help='The number of clients.')


def model(args: Namespace) -> 'Collection[Agent]':
    '''
    Create a model given the command line flags.
    '''
    if args.partial:
        server = PartialServerAgent
    else:
        server = FullServerAgent

    if args.invalid:
        global SERVER_STATE
        SERVER_STATE = InvalidServerState

    return [server.new('server')] + [ClientAgent.new('client-%s' % client_index) for client_index in range(args.clients)]


# Investigate a system with a single server and two clients.
if __name__ == '__main__':
    main(description='Simple model', flags=flags, model=model)
