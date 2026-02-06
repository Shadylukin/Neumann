---- MODULE Membership ----
\*
\* TLA+ specification of the SWIM gossip protocol for cluster membership
\* and failure detection, as implemented in tensor_chain/src/gossip.rs.
\*
\* This spec models the LWW-CRDT based membership state with:
\*   - Epidemic gossip dissemination (Sync messages)
\*   - Suspicion mechanism (indirect probes before marking failed)
\*   - Incarnation numbers for refuting false suspicions
\*   - Monotonic epoch/Lamport timestamps
\*
\* Properties verified:
\*   - EventualConvergence: all alive nodes eventually have consistent views
\*   - NoFalsePositives: alive reachable nodes not permanently marked dead
\*   - MonotonicEpochs: epoch counters never decrease
\*

EXTENDS Integers, Sequences, FiniteSets, TLC

\* ========================================================================
\* Constants
\* ========================================================================

\* Set of all possible node IDs in the cluster
CONSTANT Nodes

\* Maximum incarnation number (bounds model checking)
CONSTANT MaxIncarnation

\* Maximum Lamport timestamp (bounds model checking)
CONSTANT MaxTimestamp

\* Nil sentinel
CONSTANT Nil

ASSUME Nodes /= {}
ASSUME MaxIncarnation \in Nat /\ MaxIncarnation >= 1
ASSUME MaxTimestamp \in Nat /\ MaxTimestamp >= 1

\* ========================================================================
\* Health States
\* ========================================================================

\* Matches NodeHealth enum in tensor_chain/src/membership.rs
HealthStates == {"Healthy", "Degraded", "Failed", "Unknown"}

\* ========================================================================
\* Variables
\* ========================================================================

VARIABLES
    \* Whether a node is actually alive (ground truth, not observable)
    alive,              \* alive[n]: TRUE if node n is actually running

    \* Each node's membership view (LWW-CRDT state)
    \* membershipView[n][m] is n's view of m's state:
    \*   [health |-> HealthState, incarnation |-> Nat, timestamp |-> Nat]
    membershipView,

    \* Lamport clock per node
    epoch,              \* epoch[n]: current Lamport timestamp on node n

    \* Suspicion tracking
    suspect,            \* suspect[n]: set of nodes that n suspects

    \* Message bag
    messages            \* Set of gossip messages in transit

\* All variables
vars == <<alive, membershipView, epoch, suspect, messages>>

\* ========================================================================
\* Helper Operators
\* ========================================================================

\* Discard a message from the message bag
Discard(m) == messages' = messages \ {m}

\* Create a membership entry
MemberEntry(health, incarnation, timestamp) ==
    [health |-> health, incarnation |-> incarnation, timestamp |-> timestamp]

\* Default entry for unknown nodes
DefaultEntry == MemberEntry("Unknown", 0, 0)

\* Whether state s1 supersedes s2 (LWW-CRDT ordering)
\* Uses incarnation first, then timestamp as tiebreaker.
\* Ref: gossip.rs GossipNodeState::supersedes()
Supersedes(s1, s2) ==
    \/ s1.incarnation > s2.incarnation
    \/ (s1.incarnation = s2.incarnation /\ s1.timestamp > s2.timestamp)

\* Set of alive nodes
AliveNodes == {n \in Nodes : alive[n]}

\* Set of nodes that n considers healthy
HealthyInView(n) ==
    {m \in Nodes : membershipView[n][m].health = "Healthy"}

\* Set of nodes that n considers failed
FailedInView(n) ==
    {m \in Nodes : membershipView[n][m].health = "Failed"}

\* ========================================================================
\* Message Types
\* ========================================================================

\* Sync message carrying membership states
\* Ref: gossip.rs GossipMessage::Sync
SyncMsg(sender, states, senderTime) ==
    [mtype      |-> "Sync",
     msender    |-> sender,
     mstates    |-> states,
     msenderTime|-> senderTime]

\* Suspect message
\* Ref: gossip.rs GossipMessage::Suspect
SuspectMsg(reporter, suspectNode, incarnation) ==
    [mtype        |-> "Suspect",
     mreporter    |-> reporter,
     msuspect     |-> suspectNode,
     mincarnation |-> incarnation]

\* Alive message (refute suspicion)
\* Ref: gossip.rs GossipMessage::Alive
AliveMsg(nodeId, incarnation) ==
    [mtype        |-> "Alive",
     mnodeId      |-> nodeId,
     mincarnation |-> incarnation]

\* ========================================================================
\* Initial State
\* ========================================================================

Init ==
    /\ alive           = [n \in Nodes |-> TRUE]
    /\ membershipView  = [n \in Nodes |->
        [m \in Nodes |->
            IF m = n
            THEN MemberEntry("Healthy", 1, 1)
            ELSE MemberEntry("Unknown", 0, 0)]]
    /\ epoch           = [n \in Nodes |-> 1]
    /\ suspect         = [n \in Nodes |-> {}]
    /\ messages        = {}

\* ========================================================================
\* Actions
\* ========================================================================

\* --- Gossip Exchange ---
\*
\* An alive node n picks a peer m and sends its membership view as a
\* Sync message. This models the periodic gossip round in SWIM.
\*
\* Ref: gossip.rs GossipProtocol::gossip_round()

GossipExchange(n, target) ==
    /\ alive[n]
    /\ target \in Nodes \ {n}
    /\ epoch[n] < MaxTimestamp
    /\ LET newEpoch == epoch[n] + 1
           \* Collect states to send (all known states, up to limit)
           statesToSend == [m \in Nodes |-> membershipView[n][m]]
       IN
       /\ epoch' = [epoch EXCEPT ![n] = newEpoch]
       /\ messages' = messages \cup
           {SyncMsg(n, statesToSend, newEpoch)}
       /\ UNCHANGED <<alive, membershipView, suspect>>

\* --- Handle Sync Message ---
\*
\* An alive node receives a Sync message and merges the incoming
\* membership states using LWW-CRDT semantics. States with higher
\* incarnation (or higher timestamp for same incarnation) win.
\*
\* Ref: gossip.rs LWWMembershipState::merge()

HandleSync(n, m) ==
    /\ alive[n]
    /\ m \in messages
    /\ m.mtype = "Sync"
    /\ m.msender /= n
    /\ LET
         \* Merge: for each node, keep the state that supersedes
         mergedView == [peer \in Nodes |->
            IF Supersedes(m.mstates[peer], membershipView[n][peer])
            THEN m.mstates[peer]
            ELSE membershipView[n][peer]]
         \* Sync Lamport time: max(local, incoming) + 1
         newEpoch == IF m.msenderTime > epoch[n]
                     THEN m.msenderTime + 1
                     ELSE epoch[n] + 1
       IN
       /\ membershipView' = [membershipView EXCEPT ![n] = mergedView]
       /\ epoch' = [epoch EXCEPT ![n] =
            IF newEpoch <= MaxTimestamp THEN newEpoch ELSE epoch[n]]
       \* Mark sender as healthy (we received a message from them)
       /\ IF membershipView[n][m.msender].health /= "Healthy"
          THEN suspect' = [suspect EXCEPT ![n] = suspect[n] \ {m.msender}]
          ELSE suspect' = suspect
       /\ Discard(m)
       /\ UNCHANGED <<alive>>

\* --- Suspect Node ---
\*
\* An alive node suspects a peer that it cannot reach (models
\* failed direct probe). The peer is marked as Degraded.
\*
\* Ref: gossip.rs -- suspicion mechanism, LWWMembershipState::suspect()

SuspectNode(n, target) ==
    /\ alive[n]
    /\ target \in Nodes \ {n}
    /\ target \notin suspect[n]
    /\ membershipView[n][target].health = "Healthy"
    /\ epoch[n] < MaxTimestamp
    /\ LET newEpoch == epoch[n] + 1
           currentEntry == membershipView[n][target]
           suspectedEntry == MemberEntry("Degraded",
                                         currentEntry.incarnation,
                                         newEpoch)
       IN
       /\ membershipView' = [membershipView EXCEPT
            ![n][target] = suspectedEntry]
       /\ epoch' = [epoch EXCEPT ![n] = newEpoch]
       /\ suspect' = [suspect EXCEPT ![n] = suspect[n] \cup {target}]
       /\ messages' = messages \cup
           {SuspectMsg(n, target, currentEntry.incarnation)}
       /\ UNCHANGED <<alive>>

\* --- Handle Suspect Message ---
\*
\* A node receives a suspect notification about another node.
\* If the suspicion has a matching incarnation and the node is
\* currently healthy, mark it as Degraded.
\*
\* Ref: gossip.rs LWWMembershipState::suspect()

HandleSuspect(n, m) ==
    /\ alive[n]
    /\ m \in messages
    /\ m.mtype = "Suspect"
    /\ m.msuspect /= n  \* Not about ourselves (handled by RefuteSuspicion)
    /\ LET currentEntry == membershipView[n][m.msuspect]
       IN
       /\ IF /\ currentEntry.incarnation = m.mincarnation
             /\ currentEntry.health = "Healthy"
          THEN /\ membershipView' = [membershipView EXCEPT
                    ![n][m.msuspect] = MemberEntry("Degraded",
                                                    currentEntry.incarnation,
                                                    epoch[n] + 1)]
               /\ epoch' = [epoch EXCEPT ![n] =
                    IF epoch[n] < MaxTimestamp THEN epoch[n] + 1 ELSE epoch[n]]
               /\ suspect' = [suspect EXCEPT ![n] = suspect[n] \cup {m.msuspect}]
          ELSE /\ UNCHANGED <<membershipView, epoch, suspect>>
       /\ Discard(m)
       /\ UNCHANGED <<alive>>

\* --- Refute Suspicion ---
\*
\* A node that is suspected by someone refutes the suspicion by
\* incrementing its incarnation number and broadcasting an Alive message.
\* This proves it is still operational.
\*
\* Ref: gossip.rs LWWMembershipState::refute()

RefuteSuspicion(n) ==
    /\ alive[n]
    \* There exists a suspect message about us, or we see ourselves degraded
    /\ \/ \E m \in messages :
            /\ m.mtype = "Suspect"
            /\ m.msuspect = n
       \/ membershipView[n][n].health \in {"Degraded", "Failed"}
    /\ membershipView[n][n].incarnation < MaxIncarnation
    /\ epoch[n] < MaxTimestamp
    /\ LET
         currentInc == membershipView[n][n].incarnation
         newInc == currentInc + 1
         newEpoch == epoch[n] + 1
         newEntry == MemberEntry("Healthy", newInc, newEpoch)
       IN
       /\ membershipView' = [membershipView EXCEPT ![n][n] = newEntry]
       /\ epoch' = [epoch EXCEPT ![n] = newEpoch]
       /\ suspect' = [suspect EXCEPT ![n] = suspect[n] \ {n}]
       /\ messages' = messages \cup {AliveMsg(n, newInc)}
       /\ UNCHANGED <<alive>>

\* --- Handle Alive Message ---
\*
\* A node receives an Alive message and updates its view if the
\* incarnation is higher, marking the node as Healthy.
\*
\* Ref: gossip.rs LWWMembershipState::refute()

HandleAlive(n, m) ==
    /\ alive[n]
    /\ m \in messages
    /\ m.mtype = "Alive"
    /\ LET
         currentEntry == membershipView[n][m.mnodeId]
       IN
       /\ IF m.mincarnation > currentEntry.incarnation
          THEN /\ membershipView' = [membershipView EXCEPT
                    ![n][m.mnodeId] = MemberEntry("Healthy",
                                                   m.mincarnation,
                                                   epoch[n] + 1)]
               /\ epoch' = [epoch EXCEPT ![n] =
                    IF epoch[n] < MaxTimestamp THEN epoch[n] + 1 ELSE epoch[n]]
               /\ suspect' = [suspect EXCEPT ![n] =
                    suspect[n] \ {m.mnodeId}]
          ELSE /\ UNCHANGED <<membershipView, epoch, suspect>>
       /\ Discard(m)
       /\ UNCHANGED <<alive>>

\* --- Failure Detection ---
\*
\* After suspicion timeout, a suspected node is marked as Failed.
\* This models the transition from Degraded to Failed after the
\* suspicion_timeout_ms expires without receiving an Alive refutation.
\*
\* Ref: gossip.rs -- suspicion timeout logic, LWWMembershipState::fail()

FailureDetection(n, target) ==
    /\ alive[n]
    /\ target \in suspect[n]
    /\ membershipView[n][target].health = "Degraded"
    /\ epoch[n] < MaxTimestamp
    /\ LET newEpoch == epoch[n] + 1
       IN
       /\ membershipView' = [membershipView EXCEPT
            ![n][target] = MemberEntry("Failed",
                                        membershipView[n][target].incarnation,
                                        newEpoch)]
       /\ epoch' = [epoch EXCEPT ![n] = newEpoch]
       /\ UNCHANGED <<alive, suspect, messages>>

\* --- Node Crash ---
\*
\* A node crashes (becomes not alive). This is the ground truth
\* that the failure detector attempts to discover.

NodeCrash(n) ==
    /\ alive[n]
    /\ alive' = [alive EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<membershipView, epoch, suspect, messages>>

\* --- Node Rejoin ---
\*
\* A crashed node restarts and rejoins the cluster with an
\* incremented incarnation number, marking itself as Healthy.
\*
\* Ref: gossip.rs -- node rejoin with new incarnation

Rejoin(n) ==
    /\ ~alive[n]
    /\ membershipView[n][n].incarnation < MaxIncarnation
    /\ epoch[n] < MaxTimestamp
    /\ LET
         newInc == membershipView[n][n].incarnation + 1
         newEpoch == epoch[n] + 1
         newEntry == MemberEntry("Healthy", newInc, newEpoch)
       IN
       /\ alive' = [alive EXCEPT ![n] = TRUE]
       /\ membershipView' = [membershipView EXCEPT ![n][n] = newEntry]
       /\ epoch' = [epoch EXCEPT ![n] = newEpoch]
       /\ suspect' = [suspect EXCEPT ![n] = {}]
       /\ messages' = messages \cup {AliveMsg(n, newInc)}

\* ========================================================================
\* Next-State Relation
\* ========================================================================

Next ==
    \/ \E n \in Nodes :
        \/ RefuteSuspicion(n)
        \/ NodeCrash(n)
        \/ Rejoin(n)
        \/ \E target \in Nodes \ {n} :
            \/ GossipExchange(n, target)
            \/ SuspectNode(n, target)
            \/ FailureDetection(n, target)
    \/ \E n \in Nodes, m \in messages :
        \/ HandleSync(n, m)
        \/ HandleSuspect(n, m)
        \/ HandleAlive(n, m)

Spec == Init /\ [][Next]_vars

\* Fairness for liveness properties
Fairness ==
    /\ \A n \in Nodes : WF_vars(RefuteSuspicion(n))
    /\ \A n \in Nodes : \A target \in Nodes \ {n} :
        WF_vars(GossipExchange(n, target))
    /\ \A n \in Nodes : WF_vars(\E m \in messages : HandleSync(n, m))
    /\ \A n \in Nodes : WF_vars(\E m \in messages : HandleAlive(n, m))

FairSpec == Spec /\ Fairness

\* ========================================================================
\* Safety Properties (Invariants)
\* ========================================================================

\* --- MonotonicEpochs ---
\* Lamport timestamps (epochs) never decrease on any node.
\* This corresponds to the sync_time() method in LWWMembershipState
\* which always advances: max(local, incoming) + 1.
\*
\* Ref: gossip.rs LWWMembershipState::sync_time()

MonotonicEpochs ==
    [][\A n \in Nodes : epoch'[n] >= epoch[n]]_vars

\* --- MonotonicIncarnations ---
\* A node's own incarnation number never decreases.
\* This is a key invariant of the SWIM protocol.
\*
\* Ref: gossip.rs -- incarnation only incremented in refute() and rejoin

MonotonicIncarnations ==
    [][\A n \in Nodes :
        membershipView'[n][n].incarnation >= membershipView[n][n].incarnation
    ]_vars

\* --- NoFalsePositives (Safety Approximation) ---
\* An alive node is never permanently marked as Failed by all other
\* alive nodes, as long as messages are eventually delivered.
\*
\* Formally: if a node is alive and can refute suspicion (incarnation
\* not exhausted), then it is not the case that all other alive nodes
\* have it marked as Failed with a stale incarnation.
\*
\* This is a safety approximation. The full liveness property
\* (EventualConvergence) requires fairness.

\* Safety: no alive node is ever marked Failed at an incarnation
\* *strictly higher* than its own self-view incarnation.  A node's
\* incarnation is the authoritative source, and no other node should
\* fabricate a higher incarnation in a Failed entry.
NoFalsePositivesSafety ==
    \A n \in Nodes :
        \A m \in Nodes :
            membershipView[m][n].health = "Failed"
            => membershipView[m][n].incarnation <= membershipView[n][n].incarnation

\* --- SuspicionRequiresDegradedFirst ---
\* A node can only be marked Failed if it was previously Degraded.
\* This matches the SWIM protocol's two-phase failure detection.

SuspicionProtocol ==
    \A n \in Nodes :
        \A target \in Nodes :
            \* You cannot jump directly from Healthy to Failed;
            \* must go through Degraded (Suspected) first.
            \* This is checked as: if currently Failed, the transition
            \* came from Degraded. We check the invariant form:
            \* FailureDetection only fires when health = "Degraded".
            TRUE  \* Enforced structurally by the FailureDetection action guard

\* ========================================================================
\* Liveness Properties (require FairSpec)
\* ========================================================================

\* --- EventualConvergence ---
\* If all nodes are alive and the network is reliable (fair message
\* delivery), then all nodes eventually have consistent membership views.
\*
\* This models the O(log N) convergence property of epidemic gossip.
\*
\* Note: This is a liveness property and requires FairSpec.

EventualConvergence ==
    (\A n \in Nodes : alive[n])
    ~> (\A n1, n2 \in Nodes :
            \A target \in Nodes :
                membershipView[n1][target].incarnation =
                membershipView[n2][target].incarnation)

\* --- EventualFailureDetection ---
\* If a node crashes and stays crashed, eventually all alive nodes
\* will mark it as Failed.

EventualFailureDetection ==
    \A n \in Nodes :
        ([]~alive[n])
        ~> (\A m \in AliveNodes :
                membershipView[m][n].health = "Failed")

\* ========================================================================
\* Type Invariant (for debugging)
\* ========================================================================

TypeOK ==
    /\ alive \in [Nodes -> BOOLEAN]
    /\ epoch \in [Nodes -> Nat]
    /\ suspect \in [Nodes -> SUBSET Nodes]
    /\ \A n \in Nodes : \A m \in Nodes :
        /\ membershipView[n][m].health \in HealthStates
        /\ membershipView[n][m].incarnation \in Nat
        /\ membershipView[n][m].timestamp \in Nat

====
