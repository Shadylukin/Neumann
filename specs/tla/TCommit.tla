---- MODULE TCommit ----
\*
\* Abstract transaction commit specification (Lamport's TCommit).
\*
\* This is the refinement target for TwoPhaseCommit.tla. It models
\* the fundamental safety property of any transaction commit protocol:
\* participants independently decide to prepare, commit, or abort,
\* subject to the constraint that a commit requires all participants
\* to have prepared, and no participant can be both committed and
\* aborted.
\*
\* Ref: Lamport, "Transaction Commit" (2004)

CONSTANT Participants

VARIABLE rmState   \* rmState[p]: "working" | "prepared" | "committed" | "aborted"

\* ========================================================================
\* Safety Property
\* ========================================================================

\* No participant has committed while another has aborted.
TCConsistent ==
    \A p1, p2 \in Participants :
        ~(rmState[p1] = "committed" /\ rmState[p2] = "aborted")

\* ========================================================================
\* State Machine
\* ========================================================================

TCInit == rmState = [p \in Participants |-> "working"]

Prepare(p) ==
    /\ rmState[p] = "working"
    /\ rmState' = [rmState EXCEPT ![p] = "prepared"]

Decide(p) ==
    /\ rmState[p] = "prepared"
    /\ \A q \in Participants : rmState[q] \in {"prepared", "committed"}
    /\ rmState' = [rmState EXCEPT ![p] = "committed"]

Abort(p) ==
    /\ rmState[p] \in {"working", "prepared"}
    /\ rmState' = [rmState EXCEPT ![p] = "aborted"]

TCNext == \E p \in Participants : Prepare(p) \/ Decide(p) \/ Abort(p)

TCSpec == TCInit /\ [][TCNext]_rmState

====
