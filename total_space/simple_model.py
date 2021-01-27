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


class ParentAgent(Agent):
    '''
    A no-op parent peeking into the state of the clients.
    '''

    def __init__(self) -> None:
        '''
        Create a parent in the trivial (empty) state.
        '''
        super().__init__(name='client', state=State(name='idle', data=None))

    def _time_when_idle(self, _message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE

    def _check_when_idle(self, message: Message) -> Optional[Collection[Action]]:
        child_name = message.state.data
        assert isinstance(child_name, str)
        assert child_name.startswith('client-')
        assert isinstance(self.children[child_name], State)
        assert self.children[child_name].name == 'check'
        confirm = Message(source_agent_name=self.name, target_agent_name=child_name, state=State(name='confirm!', data=child_name))
        return [Action(name='send_confirm', next_state=None, send_messages=(confirm,))]


class ClientAgent(Agent):
    '''
    A simple client that talks with a singleton server.
    '''

    #: Trivial state for the client agent.
    IdleState = State(name='idle', data=None)

    #: Trivial state for the client agent.
    CheckState = State(name='check', data=None)

    def __init__(self, index: int) -> None:
        '''
        Create a client in the initial (idle) state.
        '''
        with initializing(self):  # Required to allow modifying this immutable while initializing it.
            self.index = index  # Can add data members which will be immutable from here on.
        super().__init__(name='client-%s' % index, state=ClientAgent.IdleState, max_in_flight_messages=2)

    def _time_when_wait(self, _message: Message) -> Optional[Collection[Action]]:
        assert isinstance(self.state.data, int)
        if self.state.data == 1:
            request = Message(source_agent_name=self.name, target_agent_name='server', state=State(name='request@', data=self.name))
            return [Action(name='send_request', next_state=State(name='wait', data=2), send_messages=(request,))]
        else:
            return Agent.IGNORE

    def _time_when_idle(self, _message: Message) -> Optional[Collection[Action]]:
        request = Message(source_agent_name=self.name, target_agent_name='server', state=State(name='request@', data=self.name))
        check = Message(source_agent_name=self.name, target_agent_name='client', state=State(name='(check)?=>check', data=self.name))
        return [Action(name='send_request', next_state=State(name='wait', data=1), send_messages=(request,)),
                Action(name='check_parent', next_state=ClientAgent.CheckState, send_messages=(check,))]

    def is_deferring(self) -> bool:
        return self.state.name == 'check'

    def _time_when_check(self, _message: Message) -> Optional[Collection[Action]]:
        recheck = Message(source_agent_name=self.name, target_agent_name='client', state=State(name='(check)?=>check', data=self.name))
        return [Action(name='recheck', next_state=None, send_messages=(recheck,))]

    def _response_when_wait(self, message: Message) -> Optional[Collection[Action]]:
        assert message.state.data == self.name
        if self.state.data == 2:
            return [Action(name='receive_response', next_state=State(name='wait', data=1), send_messages=())]
        else:
            return [Action(name='receive_response', next_state=ClientAgent.IdleState, send_messages=())]

    def _confirm_when_check(self, message: Message) -> Optional[Collection[Action]]:
        assert message.state.data == self.name
        return [Action(name='receive_confirm', next_state=ClientAgent.IdleState, send_messages=())]


class InvalidServerState(State):
    '''
    A server state that reports invalid errors, for tests.
    '''

    def validate(self) -> Collection[str]:
        if self.data == 'client-1':
            return ['client-1 is banned']
        return ()


class PartialServerAgent(Agent):
    '''
    A partial server (for testing).

    The server handles multiple clients, one at a time.
    '''

    #: Class to use for server states.
    #:
    #: In general the server state data is the client being served, if any.
    #:
    #: We therefore could have had just a single state name.
    #:
    #: Instead this uses a different name `ready` when there is no client being served, and `busy`
    #: when actually working on behalf of a client.
    #:
    #: In general it is always possible to have a single state name and just place all the information
    #: in the data field.
    #:
    #: However using clear state names makes it easier to understand the state machine logic.
    StateClass: Type[State] = State

    #: Trivial state for the server agent.
    ReadyState = State(name='ready', data=None)

    def __init__(self, clients_count: int) -> None:
        '''
        Create a server in the initial (ready) state.
        '''
        super().__init__(name='server', state=PartialServerAgent.ReadyState, max_in_flight_messages=2 * clients_count)

    def _time_when_busy(self, _message: Message) -> Optional[Collection[Action]]:
        assert isinstance(self.state.data, str)
        done_client_name = self.state.data
        response = Message(source_agent_name=self.name,
                           target_agent_name=done_client_name,
                           state=State(name='response@', data=done_client_name))

        return [Action(name='done', next_state=PartialServerAgent.ReadyState, send_messages=(response,))]

    def _time_when_ready(self, _message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED

    def _request_when_ready(self, message: Message) -> Optional[Collection[Action]]:
        next_state = PartialServerAgent.StateClass(name='busy', data=message.state.data)
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
    group.add_argument('-P', '--Partial', action='store_true', help='Generate a partial model.')
    group.add_argument('-I', '--Invalid', action='store_true', help='Generate an invalid model.')
    group.add_argument('-C', '--Clients', metavar='NUMBER', type=int, default=2, help='The number of clients.')


def model(args: Namespace) -> List[Agent]:
    '''
    Create a model given the command line flags.
    '''
    if args.Invalid:
        PartialServerAgent.StateClass = InvalidServerState
    else:
        PartialServerAgent.StateClass = State

    agents: List[Agent]
    if args.Partial:
        agents = [PartialServerAgent(args.Clients)]
    else:
        agents = [FullServerAgent(args.Clients)]

    agents += [ParentAgent()]
    agents += [ClientAgent(client_index) for client_index in range(args.Clients)]
    return agents


# Investigate a system with a single server and two clients.
if __name__ == '__main__':
    main(description='Simple model', flags=flags, model=model)
