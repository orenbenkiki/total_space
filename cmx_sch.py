
# CMX Total Space Model

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

'''
    This variant considers the app behaviour as LCPU actions.
    Another variant might consider the app as a separate agent
    which would issue instructions to send messages to the LCPU.
'''

SIG_LIMIT = 2

# Logical CPU States. Data fields are:
# 0: XREQ_SIG  - signature from last (current) XREQ
# 1: ALLOC_SIG - signature from last slow_alloc
# 2: BACK_SIG  - signature from last back

LCPU_INIT_STATE   = State(name='lcpu_noreq',   data=[SIG_LIMIT-1, SIG_LIMIT-1, SIG_LIMIT-1])
# LCPU_NOREQ_STATE   = State(name='lcpu_noreq',   data=0)
# LCPU_REQ_STATE     = State(name='lcpu_req',     data=0)
# LCPU_PEND_STATE    = State(name='lcpu_pend',    data=0)
# LCPU_GRANT_STATE   = State(name='lcpu_grant',   data=0)
# LCPU_WAIT_STATE    = State(name='lcpu_wait',    data=0)
# LCPU_OWN_STATE     = State(name='lcpu_own',     data=0)
# LCPU_CALL_STATE    = State(name='lcpu_call',    data=0)
# LCPU_CANCEL_STATE  = State(name='lcpu_cancel',  data=0)


class LCPU_Agent(Agent):
    # Logical CPU.

    def __init__(self, index: int) -> None:
        # Create a client in the initial (idle) state.
        with initializing(self):  							# Required to allow modifying this immutable while initializing it.
            self.index = index  						    # Can add data members which will be immutable from here on.
        super().__init__(name='LCPU-%s' % index, state=LCPU_INIT_STATE, max_in_flight_messages=3) # Time drives max of 2

    # @staticmethod
    # def new(num) -> 'LCPU_Agent':
	#
    #    # Create a logical CPU in default (NOREQ) state.
    #    return LCPU_Agent(name="LCPU_" + str(num), index=num, state=LCPU_INIT_STATE)


    def is_deferring(self) -> bool:
        return self.state.name == 'lcpu_req' or self.state.name == 'lcpu_cancel'

    # NOREQ: No outstanding requests.

    def _time_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        alloc_sig  = self.state.data[1]
        back_sig   = self.state.data[2]
        xreq_sig   = (int(back_sig)+1) % SIG_LIMIT
        xreq_msg   = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xreq@', data=xreq_sig))
        return [Action(name='issue_xreq',   next_state=State(name='lcpu_req',   data=[xreq_sig, alloc_sig, back_sig]),   send_messages=(xreq_msg,)),
                Action(name='issue_xwait',  next_state=State(name='lcpu_noreq', data=self.state.data), send_messages=()),
                Action(name='issue_cancel', next_state=State(name='lcpu_noreq', data=self.state.data), send_messages=()),        # suppressed
                Action(name='issue_xcall',  next_state=State(name='lcpu_noreq', data=self.state.data), send_messages=()),        # suppressed
                Action(name='interrupt',    next_state=State(name='lcpu_noreq', data=self.state.data), send_messages=())]
    def _resp_reject_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE                                                                         # Can happen if LCPU cancel overlaps slow grant
    def _cmx_back_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED                                                                         # CHECK

    # These are only relevant in the separate app agent case
    # def _cancel_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
    #     return Agent.IGNORE
    # def _xwait_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
    #     return Agent.IGNORE
    # def _xcall_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
    # #     return Agent.IGNORE
    # def _interrupt_when_lcpu_noreq(self, message: Message) -> Optional[Collection[Action]]:
    #     return Agent.IGNORE

    # REQ: XREQ issued, awaiting response. No instructions can issue etc.

    def _time_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE
    def _resp_reject_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        # We will not have space to put a signature in the respone to xreq & since this is a NP instruction anyway, do not check it.
        # print("DEBUG - PEND_RSP to %s in REQ:%d, ALLOC:%d" % (self.name, self.state.data[0], self.state.data[1]))
        return [Action(name='goto_pend', next_state=State(name='lcpu_pend', data=self.state.data), send_messages=())]
    def _resp_grant_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        # We will not have space to put a signature in the respone to xreq & since this is a NP instruction anyway, do not check it.
        # print("DEBUG - GRANT_RSP to %s in REQ:%d, BACK:%d" % (self.name, self.state.data[0], self.state.data[2]))
        # There is the very unlikely scenario where a back from a time based eviction passed the grant response
        # We assume this is not checked here - the XCALL should pick it up ?
        return [Action(name='goto_own', next_state=State(name='lcpu_own', data=self.state.data), send_messages=())]
    # Depending on mediator implementation, it is possible that an XREQ could get a PEND response which is delayed
    # in the fabric and passed out by the slow grant. In this case, the XWAIT would be a NOP - so move to grant state. 
    def _slow_grant_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_req', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_req(self, message: Message) -> Optional[Collection[Action]]:
        # Can occur on sequence XREQ<-resp_gnt, XREQ<-resp_pend OR: XREQ<-resp_pend, slow_alloc, XREQ<-resp_pend
        # An eviction on the first (granted) XREQ causes a BACK that returns BEFORE the response to the second XREQ.
        # The signature would NOT match current XREQ
        # But also possible if an (evict) back passed out the grant response (very unlikely) - signature would match in this case
        # Note that the BACK must always update the ALLOC signature as well as its own.
        # print("DEBUG - %s got BACK %d in lpcu_req: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_req', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]

    # PEND: Received PEND response to XREQ. Can execute code or issue XWAIT....

    def _time_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        xreq_sig    = (int(self.state.data[0])+1) % SIG_LIMIT
        alloc_sig   = self.state.data[0]
        back_sig    = self.state.data[0]
        xreq_msg    = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xreq@',    data=xreq_sig))
        xcancel_msg = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcancel@', data=self.state.data[0]))
        
        return [Action(name='issue_xreq',   next_state=State(name='lcpu_req',    data=[xreq_sig, alloc_sig, back_sig]), send_messages=(xreq_msg,)),
                Action(name='issue_xwait',  next_state=State(name='lcpu_wait',   data=self.state.data), send_messages=()),
                Action(name='issue_cancel', next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,)),
                Action(name='issue_xcall',  next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,)),      # supressed
                Action(name='interrupt',    next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,))]

    def _resp_reject_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_pend', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_pend(self, message: Message) -> Optional[Collection[Action]]:
        # Occurs if you get evicted before you issue the XWAIT - signature will be updated to match the XREQ
        # Can also occur on sequence XREQ<-resp_gnt, XREQ<-resp_pend OR: XREQ<-resp_pend, slow_alloc, XREQ<-resp_pend
        # An eviction on the first (granted) XREQ causes a BACK that returns AFTER the response to the second XREQ.
        # The signature would NOT match current XREQ in this case
        # print("DEBUG - %s got BACK %d in lpcu_pend: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_pend', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]

    # GRANT: Reached if slow grant received from CMXS while in XREQ/PEND (before XREQ completes or WAIT issued)

#    def _time_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        alloc_sig   = self.state.data[1]
#        back_sig    = self.state.data[2]
#        xreq_sig    = (cur_sig+1) % SIG_LIMIT
#        xreq_msg    = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xreq@',    data=xreq_sig))
#        xcancel_msg = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcancel@', data=self.state.data[0]))
#            return [Action(name='issue_xreq',   next_state=State(name='lcpu_req',    data=[xreq_sig, cur_sig, self.state.data[1]+1]),   send_messages=(xreq_msg,)),
#                    Action(name='issue_xwait',  next_state=State(name='lcpu_own',    data=self.state.data), send_messages=()),
#                    Action(name='issue_cancel', next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,)),
#                    Action(name='issue_xcall',  next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,)),
#                    Action(name='interrupt',    next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,))]
#    def _resp_reject_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        return Agent.UNEXPECTED
#    # This one can happen as explained above - the slow grant may have passed the PEND response
#    def _resp_pend_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        return Agent.IGNORE
#    def _resp_grant_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        return Agent.UNEXPECTED
#    def _slow_grant_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        return Agent.UNEXPECTED
#    def _cmx_back_when_lcpu_grant(self, message: Message) -> Optional[Collection[Action]]:
#        return Agent.UNEXPECTED


    # WAIT: XWAIT was issued from PEND state. In low power state. Awaiting grant from CMXS, possible timeout

    def _time_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        # print("DEBUG- WAIT with sig state XREQ:%d, ALLOC:%d, BACK:%d" % (self.state.data[0], self.state.data[1],self.state.data[2]))
        xcancel_msg = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcancel@', data=self.state.data[0]))
        if (self.state.data[0] == self.state.data[2]):              # If you have received a BACK that matches XREQ, you have been evicted....
            return[Action(name='goto_noreq', next_state=State(name='lcpu_noreq', data=[self.state.data[0], self.state.data[1], self.state.data[2]]), send_messages=())]
        elif (self.state.data[0] == self.state.data[1]):           # If you have received an ALLOC that matches XREQ, you have been allocated....
            return[Action(name='goto_own', next_state=State(name='lcpu_own', data=[self.state.data[0], self.state.data[1], self.state.data[2]]), send_messages=())]
        else:
            return [Action.NOP,
                    Action(name='xwait_timeout', next_state=State(name='lcpu_pend',   data=self.state.data), send_messages=()),
                    Action(name='interrupt',     next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,))]
    def _resp_reject_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_wait', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_wait(self, message: Message) -> Optional[Collection[Action]]:
        # print("DEBUG - %s got BACK %d in lpcu_wait: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_wait', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]


    # OWN: LCPU is aware it has been granted a COP but has yet to issue XCALL.
    def _time_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        xreq_sig    = (int(self.state.data[0])+1) % SIG_LIMIT
        alloc_sig   = self.state.data[0]
        back_sig    = self.state.data[0]
        xreq_msg    = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xreq@',    data=xreq_sig))
        xcancel_msg = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcancel@', data=self.state.data[0]))
        xcall_msg   = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcall@',   data=self.state.data[0]))

        return [Action(name='issue_xreq',   next_state=State(name='lcpu_req',    data=[xreq_sig, alloc_sig, back_sig]), send_messages=(xreq_msg,)),
                Action(name='issue_xwait',  next_state=State(name='lcpu_own',    data=self.state.data),  send_messages=()),
                Action(name='issue_cancel', next_state=State(name='lcpu_cancel', data=self.state.data),  send_messages=(xcancel_msg,)),
                Action(name='issue_xcall',  next_state=State(name='lcpu_call',   data=self.state.data),  send_messages=(xcall_msg,)),
                Action(name='interrupt',    next_state=State(name='lcpu_cancel', data=self.state.data),  send_messages=(xcancel_msg,))]
    def _resp_reject_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        # This is a stale slow grant, occuring after a fast regrant. Can be ignored.
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_own', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_own(self, message: Message) -> Optional[Collection[Action]]:
        # The CMSX can time us out if we do not issue XCALL in time. In such a case, any 
        # XCALL (or XWAIT/XCANCEL) should see the BACK and exit if signature matches.
        # print("DEBUG - %s got BACK %d in lpcu_own: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_own', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]
	

    # CALL: LCPU has issued XCALL at a point where it owns a COP.

    def _time_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        if (self.state.data[0] == self.state.data[2]):              # If you have received a BACK that matches XREQ, you have been evicted....
            return[Action(name='goto_noreq', next_state=State(name='lcpu_noreq', data=[self.state.data[0], self.state.data[1], self.state.data[2]]), send_messages=())]
        else:
            xcancel_msg = Message(source_agent_name=self.name, target_agent_name="CMXS", state=State(name='xcancel@', data=self.state.data[0]))
            return [Action.NOP,
                    Action(name='interrupt', next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=(xcancel_msg,))]
    def _resp_reject_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        # This is a stale slow grant, occuring after a fast regrant. Can be ignored.
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_call', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_call(self, message: Message) -> Optional[Collection[Action]]:
        # print("DEBUG - %s got BACK %d in lpcu_call: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_call', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]


    # CANCEL: A CANCEL was issued from any state. In low power state, awaiting response.
    # An interrupt can occur in this state, Ucode should NOT resend a cancel in such a case.

    def _time_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        if (self.state.data[0] == self.state.data[2]):              # If you have received a BACK that matches XREQ, you have been evicted....
            return[Action(name='goto_noreq', next_state=State(name='lcpu_noreq', data=[self.state.data[0], self.state.data[1], self.state.data[2]]), send_messages=())]
        else:
            return [Action.NOP,
                    Action(name='interrupt', next_state=State(name='lcpu_cancel', data=self.state.data), send_messages=())]     # Suppressed
    def _resp_reject_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_pend_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _resp_grant_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_grant_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        # This is a stale slow grant, occuring after a fast regrant. Can be ignored.
        return [Action(name='update_alloc_sig', next_state=State(name='lcpu_cancel', data=[self.state.data[0], message.state.data, self.state.data[2]]), send_messages=())]
    def _cmx_back_when_lcpu_cancel(self, message: Message) -> Optional[Collection[Action]]:
        # print("DEBUG - %s got BACK %d in lpcu_cancel: REQ:%d, BACK:%d" % (self.name, message.state.data, self.state.data[0], self.state.data[2]))
        return [Action(name='update_back_sig', next_state=State(name='lcpu_cancel', data=[self.state.data[0], message.state.data, message.state.data]), send_messages=())]


# This is the CMXS - TDB if it has real state itself
CMXS_ONLY_STATE    = State(name='cmxs_only', data=0)

class CMXS(Agent):

    def __init__(self, name: str, client_count, cop_count: int) -> None:
        # Create a client in the initial (idle) state.
        with initializing(self):  							# Required to allow modifying this immutable while initializing it.
            self.num_agents = client_count  				# Can add data members which will be immutable from here on.
            self.num_cops   = cop_count
        super().__init__(name=name, state=CMXS_ONLY_STATE, max_in_flight_messages=client_count * 2)

    # @staticmethod
    # def new(name) -> 'CMXS':
    # 	# Create a CMXS in default state.
    # 	return CMXS_Agent(name=name, state=State(name='cmxs_only', data=0), max_in_flight_messages=client_count * 2)

    def get_avail_cop(self) -> int:
        free_cop = False                        
        for j in range (self.num_cops):
            free_cop = True                        
            for k in range (self.num_agents):
                agent_fsm = "CMXS-" + str(k)
                # If any of the local agetns are in gnt/active/stop state, they own a COP, if the COP they own matches this one, it is not free
                if ((self.children[agent_fsm].name in ['cmxs_gnt', 'cmxs_active', 'cmxs_stop']) and (self.children[agent_fsm].data[0] == j)):
                    free_cop = False                     
            if free_cop:
                break
        if free_cop:
            return(j)
        else:
            return(self.num_cops)
    

    # It is easier to do slow arb here in the parent - the local FSMs can see the agent state (they know if
    # they are PEND) but they cannot see resource state - they do not know if there is a free coprocessor.
    def _time_when_cmxs_only(self, message: Message) -> Optional[Collection[Action]]:
        
        pending_agent = False
        for i in range (self.num_agents):
            local_agent_fsm = "CMXS-" + str(i)
            assert isinstance(local_agent_fsm, str)
            assert isinstance(self.children[local_agent_fsm], State)
            if (self.children[local_agent_fsm].name == 'cmxs_pend'):
                pending_agent = True
                break
        
        lcpu_agent_fsm = message.source_agent_name.replace("CMXS", "LCPU")
        alloc_cop=self.get_avail_cop()
        if (pending_agent and (alloc_cop != self.num_cops)):
            signature = self.children[local_agent_fsm].data[1]          # Not really required but we will return it anyway
            slow_alloc_msg =  Message(source_agent_name=self.name, target_agent_name=local_agent_fsm, state=State(name='slow_alloc!', data=[alloc_cop, signature]))
            return [Action.NOP,
                    Action(name='slow_arb_grant', send_messages=(slow_alloc_msg,))]
        else:               
            return Agent.IGNORE
        
        
    def _xcancel_when_cmxs_only(self, message: Message) -> Optional[Collection[Action]]:
        local_agent_fsm = message.source_agent_name.replace("LCPU", "CMXS")
        signature = message.state.data
        expected  = self.children[local_agent_fsm].data[1]
        *_, agent_index = message.source_agent_name.split("-")
        assert (signature == expected), "XCANCEL signature " + str(signature) + " did not match expected " + str(expected) + " for agent " + message.source_agent_name
        assert local_agent_fsm.startswith("CMXS-")
        assert isinstance(self.children[local_agent_fsm], State)
        fwd_cancel_msg  =  Message(source_agent_name=self.name, target_agent_name=local_agent_fsm, state=State(name='arb_cancel!', data=[0, signature]))
        return [Action(name='cancel_local', send_messages=(fwd_cancel_msg,))]
	
    def _xreq_when_cmxs_only(self, message: Message) -> Optional[Collection[Action]]:
        local_agent_fsm = message.source_agent_name.replace("LCPU", "CMXS")
        signature = message.state.data
        *_, agent_index = message.source_agent_name.split("-")
        assert local_agent_fsm.startswith("CMXS-"), local_agent_fsm
        assert isinstance(self.children[local_agent_fsm], State)
        fwd_reject_msg = Message(source_agent_name=self.name, target_agent_name=local_agent_fsm, state=State(name='arb_reject', data=signature))
        
        # # print("DEBUG: XREQ from %s with sig: %d when in: %s" % (message.source_agent_name, signature, self.children[local_agent_fsm].name))
        
        if (self.children[local_agent_fsm].name in ['cmxs_active', 'cmxs_stop']):       # Agent should not be executing in this state
            return Agent.UNEXPECTED
        elif (self.children[local_agent_fsm].name == 'cmxs_gnt'):                       # The agent already has a COP allocated, keep it
            alloc_cop = self.children[local_agent_fsm].data[0]
            fwd_grant_msg  =  Message(source_agent_name=self.name, target_agent_name=local_agent_fsm,           state=State(name='arb_grant!',  data=[alloc_cop, signature]))
            # # print("DEBUG: XREQ from %s with sig %d gets regrant %d of %d" % (message.source_agent_name, signature, alloc_cop, self.num_cops))
            return [Action(name='grant_xreq', send_messages=(fwd_grant_msg,))]
        else:
            # See if there is a resource that can be allocated
            alloc_cop=self.get_avail_cop()
            
            # # print("DEBUG: XREQ from %s with sig %d may get allocated %d of %d" % (message.source_agent_name, signature, alloc_cop, self.num_cops))
                                               
            fwd_pend_msg  =   Message(source_agent_name=self.name, target_agent_name=local_agent_fsm,           state=State(name='arb_pend!',   data=[alloc_cop, signature]))
            fwd_grant_msg  =  Message(source_agent_name=self.name, target_agent_name=local_agent_fsm,           state=State(name='arb_grant!',  data=[alloc_cop, signature]))
            if (alloc_cop == self.num_cops):                                             # Arb guaranteed to fail
                return [Action(name='queue_xreq', send_messages=(fwd_pend_msg,))]
            else:                                                                       # Arb can suceed, or fail (due to RV) 
                return [Action(name='queue_xreq', send_messages=(fwd_pend_msg,)),
                        Action(name='grant_xreq', send_messages=(fwd_grant_msg,))]
 		
    def _xcall_when_cmxs_only(self, message: Message) -> Optional[Collection[Action]]:
        # print("DEBUG - XCALL from " + message.source_agent_name + " with sig " + str(message.state.data))
        local_agent_fsm = message.source_agent_name.replace("LCPU", "CMXS")
        signature = message.state.data
        expected  = self.children[local_agent_fsm].data[1]
        *_, agent_index = message.source_agent_name.split("-")
        assert (signature == expected), "XCALL signature " + str(signature) + " did not match expected " + str(expected) + " for agent " + message.source_agent_name
        assert local_agent_fsm.startswith("CMXS-")
        assert isinstance(self.children[local_agent_fsm], State)
        alloc_cop=self.children[local_agent_fsm].data[0]                                # Not strictly required, but fwd anyway
        fwd_xcall_msg =  Message(source_agent_name=self.name, target_agent_name=local_agent_fsm, state=State(name='arb_call!',  data=[alloc_cop, signature]))
        return [Action(name='fwd_xcall', send_messages=(fwd_xcall_msg,))]
  

# This is the CMXS view of the agent state
CMXS_NULL_STATE    = State(name='cmxs_null',   data=[0, 0])
CMXS_PEND_STATE    = State(name='cmxs_pend',   data=[0, 0])
CMXS_GNT_STATE     = State(name='cmxs_gnt',    data=[0, 0])
CMXS_ACTIVE_STATE  = State(name='cmxs_active', data=[0, 0])
CMXS_STOP_STATE    = State(name='cmxs_stop',   data=[0, 0])

# There will be an FSM for each legal agent

class CMXS_Agent(Agent):

    def __init__(self, index: int) -> None:
        # Create a client in the initial (idle) state.
        with initializing(self):  							# Required to allow modifying this immutable while initializing it.
            self.index = index  							# Can add data members which will be immutable from here on.
        super().__init__(name='CMXS-%s' % index, state=CMXS_NULL_STATE, max_in_flight_messages=3)
        # Where does 6 come from: SlowG+BACK for REQ[N-1], pend_rsp+SlowG+BACK for REQ[N]
		
    # @staticmethod
    # def new(num) -> 'CMXS_Agent':
	#
    #    # Create a CMX coprocessor in default (IDLE) state.
    #    return CMXS_Agent(name="CMXS_" + str(num), index=num, state=CMXS_NULL_STATE)

    def _time_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE
    def _arb_cancel_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        # cancel_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='cmx_back@', data=self.state.data[1]))
        return Agent.IGNORE                                 # We should NOT need to send anything here. In cases where this can occur a BACK is already inflight
    def _arb_grant_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        # # print("DEBUG: Arb GRANT in NULL -> GNT")
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        grant_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_grant@', data=message.state.data[1]))
        return [Action(name='grant_xreq', next_state=State(name='cmxs_gnt',  data=message.state.data), send_messages=(grant_resp_msg,))]
    def _arb_pend_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        pend_resp_msg =   Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_pend@', data=message.state.data[1]))
        # # print("DEBUG: Arb PEND  in NULL -> GNT")
        return [Action(name='pend_xreq',  next_state=State(name='cmxs_pend', data=message.state.data), send_messages=(pend_resp_msg,))]
    def _arb_call_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE                                 # Can happen if XCALL overlaps with BACK
    def _xidle_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_alloc_when_cmxs_null(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED


    def _time_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE
    def _arb_cancel_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        cancel_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='none|slow_grant|cmx_back=>cmx_back@', data=self.state.data[1]))
        # # print("DEBUG: CANCEL triggers BACK from PEND, going to NULL")
        return [Action(name='cancel_req',   next_state=State(name='cmxs_null',  data=message.state.data),  send_messages=(cancel_resp_msg,))]
    def _arb_grant_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        grant_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_grant@',  data=message.state.data[1]))
        return [Action(name='grant_xreq', next_state=State(name='cmxs_gnt',  data=message.state.data), send_messages=(grant_resp_msg,))]
    def _arb_pend_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        pend_resp_msg =   Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_pend@', data=message.state.data[1]))
        # # print("DEBUG: Arb PEND  in NULL -> GNT")
        return [Action(name='pend_xreq',  next_state=State(name='cmxs_pend', data=message.state.data), send_messages=(pend_resp_msg,))]
    def _arb_call_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _xidle_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_alloc_when_cmxs_pend(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm = "LCPU-" + str(self.index)
        slow_grant_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm,  state=State(name='none|slow_grant|cmx_back=>slow_grant@',  data=self.state.data[1]))
        # # print("DEBUG: Slow Alloc of COP %d to %s" % (message.state.data[0], lcpu_agent_fsm))
        return [Action(name='slow_lcpu_grant', next_state=State(name='cmxs_gnt',  data=message.state.data), send_messages=(slow_grant_msg,))]

    def _time_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        back_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='none|slow_grant|cmx_back=>cmx_back@', data=self.state.data[1]))
        # print("DEBUG: Evict may trigger BACK:%d to %s from GNT, & go to NULL" %(self.state.data[1], lcpu_agent_fsm))
        return [Action.NOP,
                Action(name='eviction_before_xcall', next_state=State(name='cmxs_null',  data=self.state.data), send_messages=(back_msg,))]
    def _arb_cancel_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        cancel_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='none|slow_grant|cmx_back=>cmx_back@', data=self.state.data[1]))
        # # print("DEBUG: CANCEL triggers BACK from ACTIVE, going to NULL")
        return [Action(name='cancel_req', next_state=State(name='cmxs_null',  data=message.state.data),  send_messages=(cancel_resp_msg,))]
    def _arb_grant_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        # This is actually a regrant - but it is necessary to respond to the LCPU.
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        grant_resp_msg =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_grant@',  data=message.state.data[1]))
        return [Action(name='grant_xreq', next_state=State(name='cmxs_gnt',  data=message.state.data), send_messages=(grant_resp_msg,))]
    def _arb_pend_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        pend_resp_msg =   Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm, state=State(name='resp_pend@', data=message.state.data[1]))
        # # print("DEBUG: Arb PEND  in NULL -> GNT")
        return [Action(name='pend_xreq',  next_state=State(name='cmxs_pend', data=message.state.data), send_messages=(pend_resp_msg,))]
    def _arb_call_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        cop_alloc = "COP-" + str(self.state.data[0])
        serve_msg =  Message(source_agent_name=self.name, target_agent_name=cop_alloc, state=State(name='serve@', data=self.index))
        # print("DEBUG %s issuing a SERVE to %s at sig : %s" % (self.name, cop_alloc, self.state.data[1]))
        return [Action(name='serve_action',   next_state=State(name='cmxs_active', data=message.state.data),  send_messages=(serve_msg,))]
    def _xidle_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _slow_alloc_when_cmxs_gnt(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED

    def _time_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        cop_alloc = "COP-" + str(self.state.data[0])
        stop_msg =  Message(source_agent_name=self.name, target_agent_name=cop_alloc, state=State(name='stop@', data=self.state.data[1]))
        # # print("DEBUG: Time: %s sending stop to %s" % (self.name, cop_alloc))
        return [Action.NOP,
                Action(name='eviction_after_xcall',   next_state=State(name='cmxs_stop', data=self.state.data), send_messages=(stop_msg,))]
    def _arb_cancel_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        cop_alloc = "COP-" + str(self.state.data[0])
        stop_msg =  Message(source_agent_name=self.name, target_agent_name=cop_alloc, state=State(name='stop@', data=self.state.data[1]))
        # # print("DEBUG: Cancel: %s sending stop to %s" % (self.name, cop_alloc))
        return [Action(name='cancel_after_xcall',   next_state=State(name='cmxs_stop', data=self.state.data), send_messages=(stop_msg,))]
    def _arb_grant_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _arb_pend_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _arb_call_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _xidle_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        *_, cop_index = message.source_agent_name.split("-")
        assert (self.state.data[0] == int(cop_index)), "XIDLE returned to " + self.name + " from " + cop_index + " did not match allocated " + str(self.state.data[0])
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        back_msg        =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm,  state=State(name='none|slow_grant|cmx_back=>cmx_back@',  data=self.state.data[1]))
        # print("DEBUG: IDLE triggered BACK:%d to %s from ACTIVE, going to NULL" %(self.state.data[1], lcpu_agent_fsm))
        return [Action(name='completion', next_state=State(name='cmxs_null', data=self.state.data),  send_messages=(back_msg,))]
    def _slow_alloc_when_cmxs_active(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED

    def _time_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE
    def _arb_cancel_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE                                 # An IDLE is expected anyway so there is no required action.
    def _arb_grant_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _arb_pend_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _arb_call_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED
    def _xidle_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        *_, cop_index = message.source_agent_name.split("-")
        assert (self.state.data[0] == int(cop_index)), "XIDLE returned to " + self.name + " from " + cop_index + " did not match allocated " + str(self.state.data[0])
        lcpu_agent_fsm  = "LCPU-" + str(self.index)
        back_msg        =  Message(source_agent_name=self.name, target_agent_name=lcpu_agent_fsm,  state=State(name='none|slow_grant|cmx_back=>cmx_back@',  data=self.state.data[1]))
        # print("DEBUG: IDLE triggered BACK:%d to %s from STOP, going to NULL" %(self.state.data[1], lcpu_agent_fsm))
        return [Action(name='completion',   next_state=State(name='cmxs_null', data=self.state.data),  send_messages=(back_msg,))]
    def _slow_alloc_when_cmxs_stop(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED


#: COPROCESSOR States.
COP_INIT_STATE     = State(name='cop_idle',   data=0)
# COP_ACTIVE_STATE   = State(name='cop_active', data=0)

class COPROCESSOR(Agent):
    # A CMX compliant coprocessor.

    def __init__(self, index: int) -> None:
        # Create a client in the initial (idle) state.
        with initializing(self):  							# Required to allow modifying this immutable while initializing it.
            self.index = index  						    # Can add data members which will be immutable from here on.
        super().__init__(name='COP-%s' % index, state=COP_INIT_STATE, max_in_flight_messages=2) # Time drives max of 2

    # @staticmethod
    # def new(num) -> 'COPROCESSOR':
	#
    #    # Create a CMX coprocessor in default (IDLE) state.
    #    return COPROCESSOR(name="COP_" + str(num), index=num, state=COP_IDLE_STATE)

    # IDLE: Not owned.

    def _time_when_cop_idle(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE
    def _stop_when_cop_idle(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.IGNORE                             # Stop can be issued after job completes so this must be IGNORE
    def _serve_when_cop_idle(self, message: Message) -> Optional[Collection[Action]]:
        return [Action(name='task_init', next_state=State(name='cop_active', data=message.state.data), send_messages=())]

    # ACTIVE: Owned.

    def _time_when_cop_active(self, message: Message) -> Optional[Collection[Action]]:
        owner_name = "CMXS-"+ str(self.state.data)
        idle_msg = Message(source_agent_name=self.name, target_agent_name=owner_name, state=State(name='xidle', data=self.state.data))
        return [Action.NOP,
                Action(name='task_complete',    next_state=State(name='cop_idle', data=self.state.data), send_messages=(idle_msg,)),
                Action(name='task_exception',   next_state=State(name='cop_idle', data=self.state.data), send_messages=(idle_msg,))]
    def _stop_when_cop_active(self, message: Message) -> Optional[Collection[Action]]:
        owner_name = "CMXS-"+ str(self.state.data)
        idle_msg = Message(source_agent_name=self.name, target_agent_name=owner_name, state=State(name='xidle', data=self.state.data))
        return [Action(name='task_stopped', next_state=State(name='cop_idle', data=self.state.data), send_messages=(idle_msg,))]
    def _serve_when_cop_active(self, message: Message) -> Optional[Collection[Action]]:
        return Agent.UNEXPECTED



def log_new_configuration(configuration: Configuration) -> Collection[str]:
    print(configuration.name)
    return []

def flags(parser: ArgumentParser) -> None:
    # Add command line flags for testing partial and invalid models.
    group = parser.add_argument_group('model')
    group.add_argument('-C', '--count',     metavar='NUMBER', type=int, default=0, help='The number of agents.')
    group.add_argument('-R', '--resources', metavar='NUMBER', type=int, default=1, help='The number of resources.')


def model(args: Namespace) -> List[Agent]:

    # Create a model given the command line flags.
    agents: List[Agent] = [CMXS("CMXS", args.count, args.resources)]

    agents += [LCPU_Agent(i)  for i in range(args.count)]
    agents += [CMXS_Agent(i)  for i in range(args.count)]
    agents += [COPROCESSOR(i) for i in range(args.resources)]
	
    # Superseded by the "Agents" command
    # print("\nCreated Agents:")
    # for A in agents:
    # 	print(A.name)
    # print("\n\n")
    
    return agents

# Investigate a system.
if __name__ == '__main__':
    main(description='CMX model', flags=flags, model=model)
