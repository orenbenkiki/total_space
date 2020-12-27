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
# pylint: disable=unsubscriptable-object
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

    def _time_when_wait(self, _message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE

    def _time_when_idle(self, _message: Message) -> Optional[Collection[Action]]:
        request = Message(source_agent_name=self.name, target_agent_name='server', state=State(name='request', data=self.name))
        return [Action(name='send_request', next_state=CLIENT_WAIT_STATE, send_messages=(request,))]

    def _response_when_wait(self, message: Message) -> Optional[Collection[Action]]:
        assert message.state.data == self.name
        return [Action(name='receive_response', next_state=CLIENT_IDLE_STATE, send_messages=())]


class InvalidServerState(State):
    '''
    A server state that reports invalid errors, for tests.
    '''

    def validate(self) -> Collection[str]:
        if self.data == 'client-1':
            return ['client-1 is banned']
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
SERVER_READY_STATE = State(name='ready', data=None)


#: Class to use for server states.
SERVER_STATE: Type[State] = State


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

    def _time_when_busy(self, _message: Message) -> Optional[Collection[Action]]:
        assert isinstance(self.state.data, str)
        done_client_name = self.state.data
        response = Message(source_agent_name=self.name,
                           target_agent_name=done_client_name,
                           state=State(name='response', data=done_client_name))

        return [Action(name='done', next_state=SERVER_READY_STATE, send_messages=(response,))]

    def _time_when_ready(self, _message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED

    def _request_when_ready(self, message: Message) -> Optional[Collection[Action]]:
        next_state = SERVER_STATE(name='busy', data=message.state.data)
        return [Action(name='receive_request', next_state=next_state, send_messages=())]

    def is_deferring(self) -> bool:
        return self.state.name == 'busy'

    # Implied because ``is_deferring`` returns ``True`` in the ``busy`` state:
    #
    # def _request_when_busy(self, _message: Message) -> Optional[Collection[Action]]:
    #     return Agent.DEFER

class FullServerAgent(PartialServerAgent):
    '''
    A full server.
    '''

    def _time_when_ready(self, _message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE


def flags(parser: ArgumentParser) -> None:
    '''
    Add command line flags for testing partial and invalid models.
    '''
    group = parser.add_argument_group('model')
    group.add_argument('-p', '--partial', action='store_true', help='Generate a partial model.')
    group.add_argument('-i', '--invalid', action='store_true', help='Generate an invalid model.')
    group.add_argument('-c', '--clients', metavar='NUMBER', type=int, default=2, help='The number of clients.')


def model(args: Namespace) -> List[Agent]:
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
