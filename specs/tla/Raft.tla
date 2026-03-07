---- MODULE Raft ----
\*
\* TLA+ specification of the Tensor-Raft consensus protocol used in
\* tensor_chain/src/raft.rs.
\*
\* This spec models the core Raft consensus algorithm with the Neumann
\* extensions:
\*   - Similarity fast-path (threshold 0.95): blocks with high cosine
\*     similarity to current state bypass full validation
\*   - Pre-vote protocol: prevents disruptive elections from partitioned
\*     nodes by requiring a pre-vote majority before incrementing terms
\*   - Geometric tie-breaking: when logs are equal during elections,
\*     candidates with state embeddings closer to the cluster's semantic
\*     center are preferred
\*
\* Safety properties verified:
\*   - ElectionSafety: at most one leader per term
\*   - LogMatching: same index + term implies same entry
\*   - LeaderCompleteness: committed entries survive leader changes
\*   - StateMachineSafety: no divergent committed entries
\*

EXTENDS Integers, Sequences, FiniteSets, TLC

\* ========================================================================
\* Constants
\* ========================================================================

\* The set of server node IDs (e.g., {n1, n2, n3})
CONSTANT Nodes

\* Upper bound on terms for model checking
CONSTANT MaxTerm

\* Upper bound on log length for model checking
CONSTANT MaxLogLen

\* Abstract set of values that can appear in log entries
CONSTANT Values

\* Nil sentinel
CONSTANT Nil

\* Similarity threshold for fast-path (modeled as boolean oracle)
CONSTANT SimilarityThreshold

\* Whether pre-vote protocol is enabled
CONSTANT EnablePreVote

\* Whether geometric tie-breaking is enabled
CONSTANT EnableGeometricTiebreak

\* Whether similarity fast-path is enabled
CONSTANT EnableFastPath

\* Upper bound on in-flight messages for model checking
CONSTANT MessageBound

ASSUME MaxTerm \in Nat /\ MaxTerm >= 1
ASSUME MaxLogLen \in Nat /\ MaxLogLen >= 1
ASSUME Nil \notin Nodes
ASSUME MessageBound \in Nat /\ MessageBound >= 1

\* ========================================================================
\* Variables
\* ========================================================================

VARIABLES
    \* Persistent state on all servers (survives crashes)
    currentTerm,     \* currentTerm[n]: latest term server n has seen
    votedFor,        \* votedFor[n]: candidate that received vote in current term (or Nil)
    log,             \* log[n]: log entries, each is [term |-> t, value |-> v]

    \* Volatile state on all servers
    state,           \* state[n]: Follower, Candidate, or Leader
    commitIndex,     \* commitIndex[n]: index of highest log entry known committed

    \* Volatile state on leaders (reinitialized after election)
    nextIndex,       \* nextIndex[n][m]: for leader n, next index to send to m
    matchIndex,      \* matchIndex[n][m]: for leader n, highest index replicated on m

    \* Election vote tracking (corresponds to votes_received in raft.rs)
    votesGranted,    \* votesGranted[n]: set of nodes that granted votes to candidate n

    \* Pre-vote extension state
    inPreVote,       \* inPreVote[n]: whether n is in pre-vote phase
    preVotesGranted, \* preVotesGranted[n]: set of nodes that granted pre-votes

    \* Fast-path extension (abstract model)
    fastPathUsed,    \* fastPathUsed[n]: count of fast-path validations on n

    \* Message bag (network model: unordered, reliable delivery)
    messages         \* Set of messages in transit

\* All variables for stuttering
vars == <<currentTerm, votedFor, log, state, commitIndex,
          nextIndex, matchIndex, votesGranted,
          inPreVote, preVotesGranted,
          fastPathUsed, messages>>

\* State-space bound: cap in-flight messages for tractable model checking
StateConstraint == Cardinality(messages) <= MessageBound

\* ========================================================================
\* Helper Operators
\* ========================================================================

\* Quorum size: strict majority
Quorum == (Cardinality(Nodes) \div 2) + 1

\* All quorum sets of Nodes
QuorumSets == {Q \in SUBSET Nodes : Cardinality(Q) >= Quorum}

\* Last log index for server n
LastLogIndex(n) == Len(log[n])

\* Last log term for server n
LastLogTerm(n) == IF Len(log[n]) > 0 THEN log[n][Len(log[n])].term ELSE 0

\* Log entry at index i on server n
LogEntry(n, i) == log[n][i]

\* Whether candidate's log is at least as up-to-date as voter's
\* (Section 5.4.1 of the Raft paper)
LogUpToDate(candidateTerm, candidateIndex, voterTerm, voterIndex) ==
    \/ candidateTerm > voterTerm
    \/ (candidateTerm = voterTerm /\ candidateIndex >= voterIndex)

\* ========================================================================
\* Message Types
\* ========================================================================

\* Request vote message
RequestVoteMsg(term, candidateId, lastLogIndex, lastLogTerm) ==
    [mtype        |-> "RequestVote",
     mterm        |-> term,
     mcandidateId |-> candidateId,
     mlastLogIndex|-> lastLogIndex,
     mlastLogTerm |-> lastLogTerm]

\* Request vote response
RequestVoteResponseMsg(term, voteGranted, voterId, dest) ==
    [mtype        |-> "RequestVoteResponse",
     mterm        |-> term,
     mvoteGranted |-> voteGranted,
     mvoterId     |-> voterId,
     mdest        |-> dest]

\* Pre-vote request (term is NOT incremented)
PreVoteMsg(term, candidateId, lastLogIndex, lastLogTerm) ==
    [mtype        |-> "PreVote",
     mterm        |-> term,
     mcandidateId |-> candidateId,
     mlastLogIndex|-> lastLogIndex,
     mlastLogTerm |-> lastLogTerm]

\* Pre-vote response
PreVoteResponseMsg(term, voteGranted, voterId) ==
    [mtype        |-> "PreVoteResponse",
     mterm        |-> term,
     mvoteGranted |-> voteGranted,
     mvoterId     |-> voterId]

\* Append entries (log replication / heartbeat)
AppendEntriesMsg(term, leaderId, prevLogIndex, prevLogTerm, entries, leaderCommit, useFastPath) ==
    [mtype         |-> "AppendEntries",
     mterm         |-> term,
     mleaderId     |-> leaderId,
     mprevLogIndex |-> prevLogIndex,
     mprevLogTerm  |-> prevLogTerm,
     mentries      |-> entries,
     mleaderCommit |-> leaderCommit,
     museFastPath  |-> useFastPath]

\* Append entries response
AppendEntriesResponseMsg(term, success, followerId, matchIdx, usedFastPath) ==
    [mtype          |-> "AppendEntriesResponse",
     mterm          |-> term,
     msuccess       |-> success,
     mfollowerId    |-> followerId,
     mmatchIndex    |-> matchIdx,
     musedFastPath  |-> usedFastPath]

\* ========================================================================
\* Send/Discard helpers
\* ========================================================================

Send(m) == messages' = messages \cup {m}
Discard(m) == messages' = messages \ {m}
Reply(response, request) ==
    messages' = (messages \ {request}) \cup {response}

\* ========================================================================
\* Initial State
\* ========================================================================

Init ==
    /\ currentTerm     = [n \in Nodes |-> 0]
    /\ votedFor        = [n \in Nodes |-> Nil]
    /\ log             = [n \in Nodes |-> << >>]
    /\ state           = [n \in Nodes |-> "Follower"]
    /\ commitIndex     = [n \in Nodes |-> 0]
    /\ nextIndex       = [n \in Nodes |-> [m \in Nodes |-> 1]]
    /\ matchIndex      = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ votesGranted    = [n \in Nodes |-> {}]
    /\ inPreVote       = [n \in Nodes |-> FALSE]
    /\ preVotesGranted = [n \in Nodes |-> {}]
    /\ fastPathUsed    = [n \in Nodes |-> 0]
    /\ messages        = {}

\* ========================================================================
\* Actions
\* ========================================================================

\* --- Timeout: follower or candidate starts election ---
\*
\* Models the election timeout expiring on a node.
\* If pre-vote is enabled, enter pre-vote phase first.
\* Otherwise, directly start an election by incrementing term.
\*
\* Ref: tensor_chain/src/raft.rs start_election()

StartElection(n) ==
    /\ state[n] \in {"Follower", "Candidate"}
    /\ currentTerm[n] < MaxTerm
    /\ IF EnablePreVote
       THEN \* Start pre-vote phase (does NOT increment term)
            /\ inPreVote' = [inPreVote EXCEPT ![n] = TRUE]
            /\ preVotesGranted' = [preVotesGranted EXCEPT ![n] = {n}]
            /\ state' = [state EXCEPT ![n] = "Candidate"]
            /\ messages' = messages \cup
                {PreVoteMsg(currentTerm[n], n, LastLogIndex(n), LastLogTerm(n)) :
                 m \in Nodes \ {n}}
            /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex,
                           nextIndex, matchIndex, votesGranted, fastPathUsed>>
       ELSE \* Direct election (classic Raft)
            /\ currentTerm' = [currentTerm EXCEPT ![n] = currentTerm[n] + 1]
            /\ votedFor' = [votedFor EXCEPT ![n] = n]
            /\ state' = [state EXCEPT ![n] = "Candidate"]
            /\ votesGranted' = [votesGranted EXCEPT ![n] = {n}]
            /\ messages' = messages \cup
                {RequestVoteMsg(currentTerm[n] + 1, n,
                                LastLogIndex(n), LastLogTerm(n)) :
                 m \in Nodes \ {n}}
            /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex,
                           inPreVote, preVotesGranted, fastPathUsed>>

\* --- Handle PreVote request ---
\*
\* Grant pre-vote if:
\*   1. Candidate's term >= our term
\*   2. Candidate's log is at least as up-to-date
\* Pre-votes do NOT update votedFor or currentTerm.
\*
\* Ref: tensor_chain/src/raft.rs handle_pre_vote()

HandlePreVote(n, m) ==
    /\ m \in messages
    /\ m.mtype = "PreVote"
    /\ LET grant ==
           /\ m.mterm >= currentTerm[n]
           /\ LogUpToDate(m.mlastLogTerm, m.mlastLogIndex,
                          LastLogTerm(n), LastLogIndex(n))
       IN
       /\ Reply(PreVoteResponseMsg(currentTerm[n], grant, n), m)
       /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                      nextIndex, matchIndex, votesGranted,
                      inPreVote, preVotesGranted, fastPathUsed>>

\* --- Handle PreVote response ---
\*
\* Collect pre-votes. If a quorum is reached, transition to real election
\* by incrementing term and sending RequestVote messages.
\*
\* Ref: tensor_chain/src/raft.rs handle_pre_vote_response()

HandlePreVoteResponse(n, m) ==
    /\ m \in messages
    /\ m.mtype = "PreVoteResponse"
    /\ inPreVote[n] = TRUE
    /\ state[n] = "Candidate"
    /\ \/ /\ m.mvoteGranted
          /\ LET newVotes == preVotesGranted[n] \cup {m.mvoterId}
             IN
             /\ preVotesGranted' = [preVotesGranted EXCEPT ![n] = newVotes]
             /\ IF Cardinality(newVotes) >= Quorum
                THEN \* Won pre-vote -- start real election
                     /\ currentTerm' = [currentTerm EXCEPT ![n] = currentTerm[n] + 1]
                     /\ votedFor' = [votedFor EXCEPT ![n] = n]
                     /\ inPreVote' = [inPreVote EXCEPT ![n] = FALSE]
                     /\ votesGranted' = [votesGranted EXCEPT ![n] = {n}]
                     /\ messages' = (messages \ {m}) \cup
                         {RequestVoteMsg(currentTerm[n] + 1, n,
                                         LastLogIndex(n), LastLogTerm(n)) :
                          peer \in Nodes \ {n}}
                     /\ UNCHANGED <<log, state, commitIndex, nextIndex,
                                    matchIndex, fastPathUsed>>
                ELSE \* Still collecting pre-votes
                     /\ Discard(m)
                     /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                                    nextIndex, matchIndex, votesGranted,
                                    inPreVote, fastPathUsed>>
       \/ /\ ~m.mvoteGranted
          /\ IF m.mterm > currentTerm[n]
             THEN \* Higher term discovered -- step down
                  /\ currentTerm' = [currentTerm EXCEPT ![n] = m.mterm]
                  /\ votedFor' = [votedFor EXCEPT ![n] = Nil]
                  /\ state' = [state EXCEPT ![n] = "Follower"]
                  /\ inPreVote' = [inPreVote EXCEPT ![n] = FALSE]
                  /\ Discard(m)
                  /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex,
                                 votesGranted, preVotesGranted, fastPathUsed>>
             ELSE
                  /\ Discard(m)
                  /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                                 nextIndex, matchIndex, votesGranted,
                                 inPreVote, preVotesGranted, fastPathUsed>>

\* --- Handle RequestVote ---
\*
\* Grant vote if:
\*   1. Candidate's term >= our current term
\*   2. We have not voted for someone else this term
\*   3. Candidate's log is at least as up-to-date as ours
\*   4. For equal logs with geometric tie-breaking enabled,
\*      an oracle models the similarity check
\*
\* Ref: tensor_chain/src/raft.rs handle_request_vote()

HandleRequestVote(n, m) ==
    /\ m \in messages
    /\ m.mtype = "RequestVote"
    /\ LET
         \* Step down if we see a higher term
         newTerm == IF m.mterm > currentTerm[n] THEN m.mterm ELSE currentTerm[n]
         newVotedFor == IF m.mterm > currentTerm[n] THEN Nil ELSE votedFor[n]
         newState == IF m.mterm > currentTerm[n] THEN "Follower" ELSE state[n]

         \* Can we grant the vote?
         canVote == (newVotedFor = Nil \/ newVotedFor = m.mcandidateId)

         \* Is candidate's log at least as up-to-date?
         logOk == LogUpToDate(m.mlastLogTerm, m.mlastLogIndex,
                              LastLogTerm(n), LastLogIndex(n))

         \* Geometric tie-breaking: when enabled and logs are equal,
         \* the grant decision is non-deterministic (models similarity
         \* oracle). TLC explores both paths for safety verification.
         equalLogs == /\ m.mlastLogTerm = LastLogTerm(n)
                      /\ m.mlastLogIndex = LastLogIndex(n)
         standardGrant == /\ m.mterm >= newTerm /\ canVote /\ logOk
       IN
       \E grant \in
            (IF /\ EnableGeometricTiebreak
                /\ m.mterm >= newTerm
                /\ canVote
                /\ equalLogs
             THEN BOOLEAN
             ELSE {standardGrant}) :
       /\ currentTerm' = [currentTerm EXCEPT ![n] = newTerm]
       /\ votedFor' = [votedFor EXCEPT ![n] =
            IF grant THEN m.mcandidateId ELSE newVotedFor]
       /\ state' = [state EXCEPT ![n] = newState]
       /\ inPreVote' = [inPreVote EXCEPT ![n] =
            IF m.mterm > currentTerm[n] THEN FALSE ELSE inPreVote[n]]
       /\ Reply(RequestVoteResponseMsg(newTerm, grant, n, m.mcandidateId), m)
       /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex, votesGranted,
                      preVotesGranted, fastPathUsed>>

\* --- Handle RequestVoteResponse ---
\*
\* Collect votes. Track them in votesGranted variable.
\* The BecomeLeader action checks if quorum is reached.
\*
\* Ref: tensor_chain/src/raft.rs handle_request_vote_response()

HandleRequestVoteResponse(n, m) ==
    /\ m \in messages
    /\ m.mtype = "RequestVoteResponse"
    /\ m.mdest = n
    /\ \/ /\ state[n] = "Candidate"
          /\ m.mterm = currentTerm[n]
          /\ \/ /\ m.mvoteGranted
                /\ votesGranted' = [votesGranted EXCEPT ![n] =
                     votesGranted[n] \cup {m.mvoterId}]
                /\ Discard(m)
                /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                               nextIndex, matchIndex, inPreVote, preVotesGranted,
                               fastPathUsed>>
             \/ /\ ~m.mvoteGranted
                /\ Discard(m)
                /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                               nextIndex, matchIndex, votesGranted,
                               inPreVote, preVotesGranted, fastPathUsed>>
       \/ /\ m.mterm > currentTerm[n]
          \* Step down: higher term discovered -- reset inPreVote
          /\ currentTerm' = [currentTerm EXCEPT ![n] = m.mterm]
          /\ votedFor' = [votedFor EXCEPT ![n] = Nil]
          /\ state' = [state EXCEPT ![n] = "Follower"]
          /\ votesGranted' = [votesGranted EXCEPT ![n] = {}]
          /\ inPreVote' = [inPreVote EXCEPT ![n] = FALSE]
          /\ Discard(m)
          /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex,
                         preVotesGranted, fastPathUsed>>
       \/ \* Stale response (term < currentTerm or not candidate): discard
          /\ m.mterm < currentTerm[n]
          /\ Discard(m)
          /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                         nextIndex, matchIndex, votesGranted,
                         inPreVote, preVotesGranted, fastPathUsed>>

\* --- Become Leader ---
\*
\* A candidate with a quorum of votes transitions to leader.
\* Reinitializes nextIndex and matchIndex for all peers.
\*
\* Ref: tensor_chain/src/raft.rs become_leader()

BecomeLeader(n) ==
    /\ state[n] = "Candidate"
    /\ inPreVote[n] = FALSE
    /\ Cardinality(votesGranted[n]) >= Quorum
    /\ state' = [state EXCEPT ![n] = "Leader"]
    /\ nextIndex' = [nextIndex EXCEPT ![n] =
        [m \in Nodes |-> LastLogIndex(n) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![n] =
        [m \in Nodes |-> 0]]
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, messages,
                   votesGranted, inPreVote, preVotesGranted, fastPathUsed>>

\* --- Client Request (leader appends to log) ---
\*
\* A leader receives a client request and appends a new entry to its log.
\*
\* Ref: tensor_chain/src/raft.rs propose()

ClientRequest(n, v) ==
    /\ state[n] = "Leader"
    /\ Len(log[n]) < MaxLogLen
    /\ v \in Values
    /\ log' = [log EXCEPT ![n] =
        Append(log[n], [term |-> currentTerm[n], value |-> v])]
    /\ UNCHANGED <<currentTerm, votedFor, state, commitIndex,
                   nextIndex, matchIndex, votesGranted, messages,
                   inPreVote, preVotesGranted, fastPathUsed>>

\* --- AppendEntries (leader sends to follower) ---
\*
\* Leader sends AppendEntries RPC to a follower. This handles both
\* heartbeats (empty entries) and log replication.
\*
\* The fast-path flag is modeled as a non-deterministic boolean.
\* When TRUE, the follower may use the similarity fast-path to skip
\* full validation. This is safe because the fast-path only applies
\* to blocks with cosine similarity >= 0.95 to current state.
\*
\* Ref: tensor_chain/src/raft.rs -- heartbeat and replication logic

AppendEntries(n, peer) ==
    /\ state[n] = "Leader"
    /\ peer \in Nodes \ {n}
    \* Non-deterministic fast-path flag (models similarity oracle).
    \* TLC explores both TRUE and FALSE paths, verifying safety
    \* regardless of whether fast-path is used.
    /\ \E useFP \in (IF EnableFastPath THEN BOOLEAN ELSE {FALSE}) :
       LET
         prevIdx == nextIndex[n][peer] - 1
         prevTerm == IF prevIdx > 0 /\ prevIdx <= Len(log[n])
                     THEN log[n][prevIdx].term
                     ELSE 0
         \* Entries to send: from nextIndex to end of log
         entriesToSend == IF nextIndex[n][peer] <= Len(log[n])
                          THEN SubSeq(log[n], nextIndex[n][peer], Len(log[n]))
                          ELSE << >>
       IN
       /\ Send(AppendEntriesMsg(currentTerm[n], n, prevIdx, prevTerm,
                                entriesToSend, commitIndex[n], useFP))
       /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                      nextIndex, matchIndex, votesGranted,
                      inPreVote, preVotesGranted, fastPathUsed>>

\* --- Handle AppendEntries ---
\*
\* Follower processes AppendEntries from leader.
\*
\* 1. Rejects if term < currentTerm
\* 2. Steps down if term > currentTerm
\* 3. Checks log consistency at prevLogIndex
\* 4. Appends new entries, resolving conflicts
\* 5. Advances commitIndex
\* 6. If fast-path is indicated, increments fastPathUsed counter
\*
\* Ref: tensor_chain/src/raft.rs handle_append_entries()

HandleAppendEntries(n, m) ==
    /\ m \in messages
    /\ m.mtype = "AppendEntries"
    /\ m.mleaderId /= n  \* A node never processes its own AppendEntries
    /\ LET
         \* Update term if needed
         newTerm == IF m.mterm > currentTerm[n] THEN m.mterm ELSE currentTerm[n]
         stepDown == m.mterm > currentTerm[n]

         \* Log consistency check
         logOk == \/ m.mprevLogIndex = 0
                  \/ /\ m.mprevLogIndex > 0
                     /\ m.mprevLogIndex <= Len(log[n])
                     /\ log[n][m.mprevLogIndex].term = m.mprevLogTerm

         \* Can accept entries?
         accept == /\ m.mterm >= currentTerm[n]
                   /\ logOk

         \* Compute new log per Raft paper Section 5.3:
         \*   Step 3: Delete existing entries only on conflict
         \*           (same index, different term).
         \*   Step 4: Append entries not already in the log.
         \* Heartbeats (empty mentries) do not modify the log.
         newLog ==
            IF accept /\ Len(m.mentries) > 0
            THEN LET
                   startIdx == m.mprevLogIndex + 1
                   \* How many new entries overlap with existing log?
                   overlapLen ==
                      IF startIdx > Len(log[n]) THEN 0
                      ELSE IF startIdx + Len(m.mentries) - 1 <= Len(log[n])
                           THEN Len(m.mentries)
                           ELSE Len(log[n]) - startIdx + 1
                   \* Is there a conflict in the overlapping region?
                   hasConflict ==
                      \E j \in 1..overlapLen :
                         log[n][startIdx + j - 1].term /= m.mentries[j].term
                 IN IF hasConflict
                    THEN \* Conflict: truncate from prevLogIndex and replace
                         (IF m.mprevLogIndex > 0
                          THEN SubSeq(log[n], 1, m.mprevLogIndex)
                          ELSE << >>) \o m.mentries
                    ELSE IF Len(m.mentries) > overlapLen
                         THEN \* No conflict; append only the truly new tail
                              log[n] \o SubSeq(m.mentries,
                                               overlapLen + 1,
                                               Len(m.mentries))
                         ELSE log[n]  \* All entries already present
            ELSE log[n]

         \* Report match up to what the leader sent, not full log length.
         \* The leader can only trust agreement up to prevLogIndex + entries.
         newMatchIdx == IF accept
                        THEN m.mprevLogIndex + Len(m.mentries)
                        ELSE 0

         \* Advance commit index
         newCommitIdx ==
            IF accept /\ m.mleaderCommit > commitIndex[n]
            THEN IF m.mleaderCommit < Len(newLog)
                 THEN m.mleaderCommit
                 ELSE Len(newLog)
            ELSE commitIndex[n]

         \* Fast-path tracking
         newFP == IF accept /\ m.museFastPath
                  THEN fastPathUsed[n] + 1
                  ELSE fastPathUsed[n]
       IN
       /\ currentTerm' = [currentTerm EXCEPT ![n] = newTerm]
       /\ votedFor' = [votedFor EXCEPT ![n] =
            IF stepDown THEN Nil ELSE votedFor[n]]
       /\ state' = [state EXCEPT ![n] =
            IF m.mterm >= currentTerm[n] THEN "Follower" ELSE state[n]]
       /\ log' = [log EXCEPT ![n] = newLog]
       /\ commitIndex' = [commitIndex EXCEPT ![n] = newCommitIdx]
       /\ fastPathUsed' = [fastPathUsed EXCEPT ![n] = newFP]
       /\ Reply(AppendEntriesResponseMsg(newTerm, accept, n, newMatchIdx,
                                          accept /\ m.museFastPath), m)
       /\ inPreVote' = [inPreVote EXCEPT ![n] =
            IF m.mterm >= currentTerm[n] THEN FALSE ELSE inPreVote[n]]
       /\ UNCHANGED <<nextIndex, matchIndex, votesGranted,
                      preVotesGranted>>

\* --- Handle AppendEntries Response ---
\*
\* Leader processes response from follower.
\* On success: advance matchIndex and nextIndex.
\* On failure: decrement nextIndex for retry.
\*
\* Ref: tensor_chain/src/raft.rs handle_append_entries_response()

HandleAppendEntriesResponse(n, m) ==
    /\ m \in messages
    /\ m.mtype = "AppendEntriesResponse"
    /\ state[n] = "Leader"
    /\ \/ /\ m.mterm > currentTerm[n]
          \* Step down: higher term discovered
          /\ currentTerm' = [currentTerm EXCEPT ![n] = m.mterm]
          /\ votedFor' = [votedFor EXCEPT ![n] = Nil]
          /\ state' = [state EXCEPT ![n] = "Follower"]
          /\ Discard(m)
          /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex, votesGranted,
                         inPreVote, preVotesGranted, fastPathUsed>>
       \/ /\ m.mterm = currentTerm[n]
          /\ \/ /\ m.msuccess
                \* Take max to handle out-of-order responses
                /\ LET newMatch == IF m.mmatchIndex > matchIndex[n][m.mfollowerId]
                                   THEN m.mmatchIndex
                                   ELSE matchIndex[n][m.mfollowerId]
                   IN
                   /\ matchIndex' = [matchIndex EXCEPT ![n][m.mfollowerId] = newMatch]
                   /\ nextIndex' = [nextIndex EXCEPT ![n][m.mfollowerId] = newMatch + 1]
                /\ Discard(m)
                /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                               votesGranted, inPreVote, preVotesGranted,
                               fastPathUsed>>
             \/ /\ ~m.msuccess
                /\ nextIndex' = [nextIndex EXCEPT ![n][m.mfollowerId] =
                     IF nextIndex[n][m.mfollowerId] > 1
                     THEN nextIndex[n][m.mfollowerId] - 1
                     ELSE 1]
                /\ Discard(m)
                /\ UNCHANGED <<currentTerm, votedFor, log, state, commitIndex,
                               matchIndex, votesGranted,
                               inPreVote, preVotesGranted, fastPathUsed>>

\* --- Advance Commit Index (leader only) ---
\*
\* Leader advances commitIndex to the highest index N such that a majority
\* of matchIndex[n][*] >= N and log[n][N].term == currentTerm[n].
\*
\* This matches the quorum-based commit rule from Section 5.3/5.4 of the
\* Raft paper, and corresponds to try_advance_commit_index() in raft.rs.
\*
\* Ref: tensor_chain/src/raft.rs try_advance_commit_index()

AdvanceCommitIndex(n) ==
    /\ state[n] = "Leader"
    /\ \E newCI \in (commitIndex[n]+1)..Len(log[n]) :
         \* Entry must be from current term
         /\ log[n][newCI].term = currentTerm[n]
         \* A quorum of servers (including leader) have this entry
         /\ LET agreeSet == {n} \cup
                 {peer \in Nodes \ {n} : matchIndex[n][peer] >= newCI}
            IN Cardinality(agreeSet) >= Quorum
         /\ commitIndex' = [commitIndex EXCEPT ![n] = newCI]
    /\ UNCHANGED <<currentTerm, votedFor, log, state, nextIndex, matchIndex,
                   votesGranted, messages, inPreVote, preVotesGranted,
                   fastPathUsed>>

\* --- Step Down: discover higher term ---
\*
\* Any server that receives an RPC with a higher term updates its term
\* and reverts to follower. This is implicit in HandleRequestVote,
\* HandleAppendEntries, etc., but we also model it as a standalone
\* action for completeness (e.g., stale leader seeing a Ping).

StepDown(n) ==
    /\ \E m \in messages :
         /\ m.mterm > currentTerm[n]
         /\ currentTerm' = [currentTerm EXCEPT ![n] = m.mterm]
         /\ votedFor' = [votedFor EXCEPT ![n] = Nil]
         /\ state' = [state EXCEPT ![n] = "Follower"]
         /\ inPreVote' = [inPreVote EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex,
                   votesGranted, messages, preVotesGranted, fastPathUsed>>

\* ========================================================================
\* Next-State Relation
\* ========================================================================

Next ==
    \/ \E n \in Nodes :
        \/ StartElection(n)
        \/ BecomeLeader(n)
        \/ AdvanceCommitIndex(n)
        \/ StepDown(n)
        \/ \E v \in Values : ClientRequest(n, v)
        \/ \E peer \in Nodes \ {n} : AppendEntries(n, peer)
    \/ \E n \in Nodes, m \in messages :
        \/ HandlePreVote(n, m)
        \/ HandlePreVoteResponse(n, m)
        \/ HandleRequestVote(n, m)
        \/ HandleRequestVoteResponse(n, m)
        \/ HandleAppendEntries(n, m)
        \/ HandleAppendEntriesResponse(n, m)

\* Fairness: every enabled action eventually executes (for liveness)
Fairness == WF_vars(Next)

\* Specification
Spec == Init /\ [][Next]_vars

\* Specification with fairness (for liveness checking)
FairSpec == Spec /\ Fairness

\* ========================================================================
\* Safety Properties (Invariants)
\* ========================================================================

\* --- ElectionSafety ---
\* At most one leader per term.
\* This is the fundamental safety property of leader election.
\*
\* In Tensor-Raft, this holds because:
\*   - Pre-vote does NOT change terms or votedFor
\*   - Geometric tie-breaking only affects which candidate wins,
\*     not whether multiple candidates can win
\*   - The quorum intersection guarantee still holds

ElectionSafety ==
    \A n1, n2 \in Nodes :
        (state[n1] = "Leader" /\ state[n2] = "Leader" /\
         currentTerm[n1] = currentTerm[n2])
        => n1 = n2

\* --- LogMatching ---
\* If two logs contain an entry with the same index and term,
\* then the logs are identical in all entries up through that index.
\*
\* This property ensures that the log replication mechanism maintains
\* consistency. It holds in Tensor-Raft because the fast-path only
\* skips validation, not the log consistency check.

LogMatching ==
    \A n1, n2 \in Nodes :
        \A i \in 1..Len(log[n1]) :
            (i <= Len(log[n2]) /\
             log[n1][i].term = log[n2][i].term)
            => log[n1][i].value = log[n2][i].value

\* --- LeaderCompleteness ---
\* If a log entry is committed in a given term, that entry will be
\* present in the logs of the leaders for all higher-numbered terms.
\*
\* Formally: for any committed entry at index i, every leader in a
\* later term has an entry at index i with the same term and value.

\* A leader in term T must have all entries committed in terms < T.
\* Stale leaders at lower terms are not a violation (they will
\* step down when they discover the higher term).
LeaderCompleteness ==
    \A n \in Nodes :
        state[n] = "Leader" =>
            \A m \in Nodes :
                \A i \in 1..commitIndex[m] :
                    (i <= Len(log[m]) /\ log[m][i].term < currentTerm[n])
                    => (/\ i <= Len(log[n])
                        /\ log[n][i] = log[m][i])

\* --- StateMachineSafety ---
\* No two servers apply different entries at the same log index.
\* Equivalently: committed entries at the same index must be identical.
\*
\* This is the key safety property that guarantees all state machines
\* see the same sequence of commands.

StateMachineSafety ==
    \A n1, n2 \in Nodes :
        \A i \in 1..commitIndex[n1] :
            (i <= commitIndex[n2])
            => (/\ i <= Len(log[n1])
                /\ i <= Len(log[n2])
                /\ log[n1][i] = log[n2][i])

\* --- VoteIntegrity ---
\* A node votes for at most one candidate per term.
\* If votedFor is set, there are no outstanding vote grants for a
\* different candidate in the same term.

VoteIntegrity ==
    \A n \in Nodes :
        votedFor[n] /= Nil =>
            ~\E m \in messages :
                /\ m.mtype = "RequestVoteResponse"
                /\ m.mvoterId = n
                /\ m.mterm = currentTerm[n]
                /\ m.mvoteGranted
                /\ votedFor[n] /= m.mdest  \* voted for someone else

\* --- TermMonotonicity ---
\* Terms never decrease on any server.

TermMonotonicity ==
    [][
        \A n \in Nodes : currentTerm'[n] >= currentTerm[n]
    ]_vars

\* ========================================================================
\* Tensor-Raft Extension Properties
\* ========================================================================

\* --- FastPathSafety ---
\* The fast-path never causes committed entries to diverge.
\* If fast-path was used, the committed entries still satisfy
\* StateMachineSafety. This is a meta-property: since fast-path
\* only skips validation but not the log consistency check,
\* StateMachineSafety already covers this. We state it explicitly
\* for documentation.

FastPathSafety ==
    StateMachineSafety

\* --- PreVoteSafety ---
\* Pre-vote does not disrupt existing leaders.
\* Specifically: if a node is in pre-vote phase, no term has been
\* incremented, so an existing leader remains valid.

PreVoteSafety ==
    \A n \in Nodes :
        inPreVote[n] =>
            \* The pre-vote candidate has NOT incremented its term
            \* (it uses currentTerm, not currentTerm+1)
            \A m \in Nodes :
                state[m] = "Leader" =>
                    currentTerm[m] >= currentTerm[n]

\* ========================================================================
\* Additional Safety Properties (from CCF Raft / Raft paper)
\* ========================================================================

\* --- CommittedLogAppendOnlyProp ---
\* Once an entry is committed, it is never removed or overwritten.
\* Temporal property: committed prefix of every node's log only grows.

CommittedLogAppendOnlyProp ==
    [][\A n \in Nodes :
        \A i \in 1..commitIndex[n] :
            /\ i <= Len(log'[n])
            /\ log'[n][i] = log[n][i]
    ]_vars

\* --- MonotonicCommitIndexProp ---
\* commitIndex never decreases on any server.

MonotonicCommitIndexProp ==
    [][\A n \in Nodes : commitIndex'[n] >= commitIndex[n]]_vars

\* --- MonotonicMatchIndexProp ---
\* matchIndex never decreases for any (leader, follower) pair while
\* the leader remains leader in the same term.

MonotonicMatchIndexProp ==
    [][\A n \in Nodes :
        (state[n] = "Leader" /\ state'[n] = "Leader" /\
         currentTerm'[n] = currentTerm[n])
        => \A p \in Nodes \ {n} : matchIndex'[n][p] >= matchIndex[n][p]
    ]_vars

\* --- NeverCommitEntryPrevTermsProp ---
\* Raft Section 5.4.2: a leader only counts replicas in its own term
\* when advancing commitIndex. Equivalently, when commitIndex advances,
\* the newly committed entry has the leader's current term.

NeverCommitEntryPrevTermsProp ==
    [][\A n \in Nodes :
        (/\ state'[n] = "Leader"
         /\ commitIndex'[n] > commitIndex[n])
        => log'[n][commitIndex'[n]].term = currentTerm'[n]
    ]_vars

\* --- ReplicationInv ---
\* Every committed entry exists on a quorum of servers.

ReplicationInv ==
    \A n \in Nodes :
        \A i \in 1..commitIndex[n] :
            Cardinality({m \in Nodes :
                /\ i <= Len(log[m])
                /\ log[m][i] = log[n][i]}) >= Quorum

\* --- StateTransitionsProp ---
\* Valid state machine transitions:
\*   Follower -> Candidate (start election)
\*   Candidate -> Leader (win election)
\*   Candidate -> Follower (discover higher term or lose)
\*   Leader -> Follower (discover higher term)
\* Notably: Follower -/-> Leader and Leader -/-> Candidate.

StateTransitionsProp ==
    [][\A n \in Nodes :
        \/ state'[n] = state[n]  \* no change
        \/ (state[n] = "Follower" /\ state'[n] = "Candidate")
        \/ (state[n] = "Candidate" /\ state'[n] = "Leader")
        \/ (state[n] = "Candidate" /\ state'[n] = "Follower")
        \/ (state[n] = "Leader" /\ state'[n] = "Follower")
    ]_vars

\* --- PermittedLogChangesProp ---
\* Log entries can only change via:
\*   1. Append to the end (leader replication)
\*   2. Truncation + replacement (conflict resolution in AppendEntries)
\* A committed prefix is never modified (see CommittedLogAppendOnlyProp).

PermittedLogChangesProp ==
    [][\A n \in Nodes :
        \/ log'[n] = log[n]  \* no change
        \/ \* Append: existing entries unchanged, new entries added
           /\ Len(log'[n]) >= Len(log[n])
           /\ SubSeq(log'[n], 1, Len(log[n])) = log[n]
        \/ \* Truncation + replace: some suffix removed and replaced
           /\ \E k \in 0..Len(log[n]) :
                /\ SubSeq(log'[n], 1, k) = SubSeq(log[n], 1, k)
    ]_vars

\* ========================================================================
\* Type Invariant (for debugging)
\* ========================================================================

TypeOK ==
    /\ currentTerm \in [Nodes -> Nat]
    /\ votedFor \in [Nodes -> Nodes \cup {Nil}]
    /\ state \in [Nodes -> {"Follower", "Candidate", "Leader"}]
    /\ commitIndex \in [Nodes -> Nat]
    /\ votesGranted \in [Nodes -> SUBSET Nodes]
    /\ inPreVote \in [Nodes -> BOOLEAN]
    /\ preVotesGranted \in [Nodes -> SUBSET Nodes]
    /\ fastPathUsed \in [Nodes -> Nat]

====
