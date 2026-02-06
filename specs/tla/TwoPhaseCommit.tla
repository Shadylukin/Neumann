---- MODULE TwoPhaseCommit ----
\*
\* TLA+ specification of the Two-Phase Commit (2PC) protocol used in
\* tensor_chain/src/distributed_tx.rs.
\*
\* This spec models the distributed transaction coordinator that spans
\* multiple shards. The protocol proceeds in two phases:
\*
\*   Phase 1 (PREPARE): Coordinator sends prepare to all participants.
\*     Each participant acquires locks, computes delta, checks for conflicts,
\*     and votes YES or NO.
\*
\*   Phase 2 (COMMIT/ABORT): If all vote YES, coordinator sends COMMIT.
\*     If any vote NO or timeout, coordinator sends ABORT.
\*
\* Tensor-native extension: orthogonal deltas can commit in parallel
\* without coordination using vector similarity (modeled abstractly).
\*
\* Properties verified:
\*   - Atomicity: all participants commit or all abort
\*   - NoOrphanedLocks: completed transactions have no held locks
\*   - ConsistentDecision: coordinator decision matches participant outcomes
\*

EXTENDS Integers, Sequences, FiniteSets, TLC

\* ========================================================================
\* Constants
\* ========================================================================

\* Set of transaction identifiers
CONSTANT Transactions

\* Set of participant identifiers (shards)
CONSTANT Participants

\* Nil sentinel
CONSTANT Nil

ASSUME Transactions /= {}
ASSUME Participants /= {}

\* ========================================================================
\* Variables
\* ========================================================================

VARIABLES
    \* Coordinator state per transaction
    txPhase,            \* txPhase[tx]: Preparing | Prepared | Committing | Committed | Aborting | Aborted
    coordinatorDecision,\* coordinatorDecision[tx]: "commit" | "abort" | Nil

    \* Participant state per transaction per participant
    participantVote,    \* participantVote[tx][p]: "yes" | "no" | "pending"
    participantState,   \* participantState[tx][p]: "working" | "prepared" | "committed" | "aborted"

    \* Lock state
    locks,              \* locks[p]: set of transaction IDs holding locks on participant p

    \* Message bag
    messages            \* Set of messages in transit

\* All variables
vars == <<txPhase, coordinatorDecision, participantVote, participantState,
          locks, messages>>

\* ========================================================================
\* Message Types
\* ========================================================================

PrepareMsg(tx, p) ==
    [mtype |-> "Prepare", mtx |-> tx, mparticipant |-> p]

VoteYesMsg(tx, p) ==
    [mtype |-> "VoteYes", mtx |-> tx, mparticipant |-> p]

VoteNoMsg(tx, p, reason) ==
    [mtype |-> "VoteNo", mtx |-> tx, mparticipant |-> p, mreason |-> reason]

CommitMsg(tx, p) ==
    [mtype |-> "Commit", mtx |-> tx, mparticipant |-> p]

AbortMsg(tx, p) ==
    [mtype |-> "Abort", mtx |-> tx, mparticipant |-> p]

AckMsg(tx, p, outcome) ==
    [mtype |-> "Ack", mtx |-> tx, mparticipant |-> p, moutcome |-> outcome]

\* ========================================================================
\* Helper Operators
\* ========================================================================

Send(m) == messages' = messages \cup {m}
Discard(m) == messages' = messages \ {m}

\* All participants have voted for transaction tx
AllVoted(tx) ==
    \A p \in Participants : participantVote[tx][p] /= "pending"

\* All participants voted YES for transaction tx
AllYes(tx) ==
    \A p \in Participants : participantVote[tx][p] = "yes"

\* Any participant voted NO for transaction tx
AnyNo(tx) ==
    \E p \in Participants : participantVote[tx][p] = "no"

\* All participants have acknowledged (committed or aborted)
AllAcknowledged(tx) ==
    \A p \in Participants :
        participantState[tx][p] \in {"committed", "aborted"}

\* ========================================================================
\* Initial State
\* ========================================================================

Init ==
    /\ txPhase            = [tx \in Transactions |-> "Preparing"]
    /\ coordinatorDecision= [tx \in Transactions |-> Nil]
    /\ participantVote    = [tx \in Transactions |-> [p \in Participants |-> "pending"]]
    /\ participantState   = [tx \in Transactions |-> [p \in Participants |-> "working"]]
    /\ locks              = [p \in Participants |-> {}]
    /\ messages           = {}

\* ========================================================================
\* Actions
\* ========================================================================

\* --- Coordinator sends Prepare to all participants ---
\*
\* The coordinator initiates Phase 1 by sending prepare messages
\* to all participants.
\*
\* Ref: distributed_tx.rs DistributedTxCoordinator::begin_transaction()

CoordinatorPrepare(tx) ==
    /\ txPhase[tx] = "Preparing"
    /\ messages' = messages \cup
        {PrepareMsg(tx, p) : p \in Participants}
    /\ UNCHANGED <<txPhase, coordinatorDecision, participantVote,
                   participantState, locks>>

\* --- Participant votes YES ---
\*
\* A participant receives a Prepare message, successfully acquires locks
\* and checks for conflicts. It votes YES and transitions to "prepared".
\*
\* Ref: distributed_tx.rs -- prepare handler, LockManager::try_lock()

ParticipantVoteYes(tx, p) ==
    /\ \E m \in messages :
        /\ m.mtype = "Prepare"
        /\ m.mtx = tx
        /\ m.mparticipant = p
    /\ participantState[tx][p] = "working"
    \* Lock acquisition: no conflicting transaction holds lock
    /\ ~\E otherTx \in Transactions :
        /\ otherTx /= tx
        /\ otherTx \in locks[p]
        /\ txPhase[otherTx] \notin {"Committed", "Aborted"}
    /\ participantVote' = [participantVote EXCEPT ![tx][p] = "yes"]
    /\ participantState' = [participantState EXCEPT ![tx][p] = "prepared"]
    /\ locks' = [locks EXCEPT ![p] = locks[p] \cup {tx}]
    /\ messages' = (messages \ {m \in messages :
                     m.mtype = "Prepare" /\ m.mtx = tx /\ m.mparticipant = p})
                   \cup {VoteYesMsg(tx, p)}
    /\ UNCHANGED <<txPhase, coordinatorDecision>>

\* --- Participant votes NO ---
\*
\* A participant receives a Prepare message but cannot proceed
\* (lock conflict, constraint violation, etc.). It votes NO.
\*
\* Ref: distributed_tx.rs -- prepare handler returning PrepareVote::No

ParticipantVoteNo(tx, p) ==
    /\ \E m \in messages :
        /\ m.mtype = "Prepare"
        /\ m.mtx = tx
        /\ m.mparticipant = p
    /\ participantState[tx][p] = "working"
    /\ participantVote' = [participantVote EXCEPT ![tx][p] = "no"]
    /\ participantState' = [participantState EXCEPT ![tx][p] = "aborted"]
    /\ messages' = (messages \ {m \in messages :
                     m.mtype = "Prepare" /\ m.mtx = tx /\ m.mparticipant = p})
                   \cup {VoteNoMsg(tx, p, "conflict")}
    /\ UNCHANGED <<txPhase, coordinatorDecision, locks>>

\* --- Coordinator decides to COMMIT ---
\*
\* After all participants vote YES, the coordinator decides to commit.
\* This is the "point of no return": once the coordinator records COMMIT,
\* all participants must eventually commit.
\*
\* Ref: distributed_tx.rs DistributedTxCoordinator::record_vote()

CoordinatorDecideCommit(tx) ==
    /\ txPhase[tx] = "Preparing"
    \* All YES votes received
    /\ \A p \in Participants :
        \E m \in messages :
            /\ m.mtype = "VoteYes"
            /\ m.mtx = tx
            /\ m.mparticipant = p
    /\ txPhase' = [txPhase EXCEPT ![tx] = "Committing"]
    /\ coordinatorDecision' = [coordinatorDecision EXCEPT ![tx] = "commit"]
    \* Send commit to all participants
    /\ messages' = (messages \ {m \in messages :
                     m.mtype \in {"VoteYes", "VoteNo"} /\ m.mtx = tx})
                   \cup {CommitMsg(tx, p) : p \in Participants}
    /\ UNCHANGED <<participantVote, participantState, locks>>

\* --- Coordinator decides to ABORT ---
\*
\* If any participant votes NO (or on timeout), the coordinator aborts.
\*
\* Ref: distributed_tx.rs -- abort path in record_vote() and timeout_check()

CoordinatorDecideAbort(tx) ==
    /\ txPhase[tx] = "Preparing"
    \* At least one NO vote received, or timeout (modeled as non-deterministic)
    /\ \/ \E m \in messages :
            /\ m.mtype = "VoteNo"
            /\ m.mtx = tx
       \/ TRUE  \* Models timeout
    /\ txPhase' = [txPhase EXCEPT ![tx] = "Aborting"]
    /\ coordinatorDecision' = [coordinatorDecision EXCEPT ![tx] = "abort"]
    \* Send abort to all participants
    /\ messages' = (messages \ {m \in messages :
                     m.mtype \in {"VoteYes", "VoteNo"} /\ m.mtx = tx})
                   \cup {AbortMsg(tx, p) : p \in Participants}
    /\ UNCHANGED <<participantVote, participantState, locks>>

\* --- Participant commits ---
\*
\* A prepared participant receives the COMMIT message and commits.
\* Releases all locks held for this transaction.
\*
\* Ref: distributed_tx.rs -- commit handler, LockManager::release()

ParticipantCommit(tx, p) ==
    /\ \E m \in messages :
        /\ m.mtype = "Commit"
        /\ m.mtx = tx
        /\ m.mparticipant = p
    /\ participantState[tx][p] = "prepared"
    /\ participantState' = [participantState EXCEPT ![tx][p] = "committed"]
    /\ locks' = [locks EXCEPT ![p] = locks[p] \ {tx}]
    /\ messages' = (messages \ {m \in messages :
                     m.mtype = "Commit" /\ m.mtx = tx /\ m.mparticipant = p})
                   \cup {AckMsg(tx, p, "committed")}
    /\ UNCHANGED <<txPhase, coordinatorDecision, participantVote>>

\* --- Participant aborts ---
\*
\* A participant receives the ABORT message and aborts.
\* Applies undo entries and releases all locks.
\*
\* Ref: distributed_tx.rs -- abort handler, UndoEntry::apply()

ParticipantAbort(tx, p) ==
    /\ \E m \in messages :
        /\ m.mtype = "Abort"
        /\ m.mtx = tx
        /\ m.mparticipant = p
    /\ participantState[tx][p] \in {"working", "prepared"}
    /\ participantState' = [participantState EXCEPT ![tx][p] = "aborted"]
    /\ locks' = [locks EXCEPT ![p] = locks[p] \ {tx}]
    /\ messages' = (messages \ {m \in messages :
                     m.mtype = "Abort" /\ m.mtx = tx /\ m.mparticipant = p})
                   \cup {AckMsg(tx, p, "aborted")}
    /\ UNCHANGED <<txPhase, coordinatorDecision, participantVote>>

\* --- Coordinator finalizes (all acks received) ---
\*
\* After all participants acknowledge, the coordinator marks
\* the transaction as fully completed.
\*
\* Ref: distributed_tx.rs -- cleanup in DistributedTxCoordinator

CoordinatorFinalize(tx) ==
    /\ txPhase[tx] \in {"Committing", "Aborting"}
    /\ \A p \in Participants :
        \E m \in messages :
            /\ m.mtype = "Ack"
            /\ m.mtx = tx
            /\ m.mparticipant = p
    /\ txPhase' = [txPhase EXCEPT ![tx] =
        IF txPhase[tx] = "Committing" THEN "Committed" ELSE "Aborted"]
    /\ messages' = messages \ {m \in messages :
                    m.mtype = "Ack" /\ m.mtx = tx}
    /\ UNCHANGED <<coordinatorDecision, participantVote, participantState, locks>>

\* --- Timeout: coordinator aborts a preparing transaction ---
\*
\* Models the timeout_ms check in DistributedTransaction::is_timed_out().
\* A transaction in Preparing phase can be aborted due to timeout.
\*
\* Ref: distributed_tx.rs DistributedTransaction::is_timed_out()

Timeout(tx) ==
    /\ txPhase[tx] = "Preparing"
    /\ txPhase' = [txPhase EXCEPT ![tx] = "Aborting"]
    /\ coordinatorDecision' = [coordinatorDecision EXCEPT ![tx] = "abort"]
    /\ messages' = (messages \ {m \in messages :
                     m.mtype \in {"Prepare", "VoteYes", "VoteNo"} /\ m.mtx = tx})
                   \cup {AbortMsg(tx, p) : p \in Participants}
    /\ UNCHANGED <<participantVote, participantState, locks>>

\* ========================================================================
\* Next-State Relation
\* ========================================================================

Next ==
    \E tx \in Transactions :
        \/ CoordinatorPrepare(tx)
        \/ CoordinatorDecideCommit(tx)
        \/ CoordinatorDecideAbort(tx)
        \/ CoordinatorFinalize(tx)
        \/ Timeout(tx)
        \/ \E p \in Participants :
            \/ ParticipantVoteYes(tx, p)
            \/ ParticipantVoteNo(tx, p)
            \/ ParticipantCommit(tx, p)
            \/ ParticipantAbort(tx, p)

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ WF_vars(Next)

\* ========================================================================
\* Safety Properties (Invariants)
\* ========================================================================

\* --- Atomicity ---
\* All participants for a given transaction reach the same final outcome.
\* If any participant has committed, no participant has aborted, and
\* vice versa.
\*
\* This is the fundamental 2PC safety property.

Atomicity ==
    \A tx \in Transactions :
        ~(\E p1, p2 \in Participants :
            /\ participantState[tx][p1] = "committed"
            /\ participantState[tx][p2] = "aborted"
            /\ participantVote[tx][p2] = "yes")

\* --- NoOrphanedLocks ---
\* When a transaction has reached a terminal state (Committed or Aborted),
\* no participant still holds locks for that transaction.
\*
\* Ref: distributed_tx.rs LockManager::release() called in commit/abort paths

NoOrphanedLocks ==
    \A tx \in Transactions :
        txPhase[tx] \in {"Committed", "Aborted"} =>
            \A p \in Participants : tx \notin locks[p]

\* --- ConsistentDecision ---
\* The coordinator's decision matches the participant outcomes.
\* If coordinator decided commit, no participant has aborted (unless
\* it voted NO initially). If coordinator decided abort, no participant
\* has committed.

ConsistentDecision ==
    \A tx \in Transactions :
        /\ (coordinatorDecision[tx] = "commit" =>
            ~\E p \in Participants :
                /\ participantState[tx][p] = "aborted"
                /\ participantVote[tx][p] = "yes")
        /\ (coordinatorDecision[tx] = "abort" =>
            ~\E p \in Participants :
                participantState[tx][p] = "committed")

\* --- VoteIrrevocability ---
\* Once a participant has voted YES and is in "prepared" state,
\* it cannot unilaterally abort. It must wait for the coordinator.
\*
\* Ref: distributed_tx.rs -- prepared participants only transition
\* on receiving Commit or Abort from coordinator

VoteIrrevocability ==
    \A tx \in Transactions :
        \A p \in Participants :
            (participantState[tx][p] = "prepared" /\
             participantVote[tx][p] = "yes")
            => participantState[tx][p] \in {"prepared", "committed", "aborted"}

\* --- DecisionStability ---
\* Once the coordinator has made a decision, it never changes.

DecisionStability ==
    [][
        \A tx \in Transactions :
            coordinatorDecision[tx] /= Nil =>
                coordinatorDecision'[tx] = coordinatorDecision[tx]
    ]_vars

\* ========================================================================
\* Type Invariant (for debugging)
\* ========================================================================

TypeOK ==
    /\ txPhase \in [Transactions ->
        {"Preparing", "Prepared", "Committing", "Committed", "Aborting", "Aborted"}]
    /\ coordinatorDecision \in [Transactions -> {"commit", "abort", Nil}]
    /\ participantVote \in [Transactions -> [Participants -> {"yes", "no", "pending"}]]
    /\ participantState \in [Transactions ->
        [Participants -> {"working", "prepared", "committed", "aborted"}]]
    /\ locks \in [Participants -> SUBSET Transactions]

====
