---- MODULE Common ----
\*
\* Shared operators used by all Neumann TLA+ specifications.
\*
\* This module provides helper operators for set operations, sequences,
\* and constants used across Raft, TwoPhaseCommit, and Membership specs.
\*

EXTENDS Integers, Sequences, FiniteSets, TLC

\* ========================================================================
\* Constants
\* ========================================================================

CONSTANT Nil  \* Sentinel value representing "no value" / uninitialized

\* ========================================================================
\* Set Operations
\* ========================================================================

\* Strict majority of a set (quorum size for Raft)
Quorum(S) == (Cardinality(S) \div 2) + 1

\* All subsets of S with exactly Quorum(S) elements
QuorumSets(S) == {Q \in SUBSET S : Cardinality(Q) = Quorum(S)}

\* Whether Q is a quorum of S
IsQuorum(Q, S) == Q \subseteq S /\ Cardinality(Q) >= Quorum(S)

\* Any two quorums of the same set must overlap
THEOREM QuorumOverlap ==
    \A S : \A Q1, Q2 \in QuorumSets(S) : Q1 \cap Q2 /= {}

\* Maximum element of a non-empty set of naturals
SetMax(S) == CHOOSE x \in S : \A y \in S : x >= y

\* Minimum element of a non-empty set of naturals
SetMin(S) == CHOOSE x \in S : \A y \in S : x <= y

\* ========================================================================
\* Sequence Operations
\* ========================================================================

\* Last element of a non-empty sequence
Last(seq) == seq[Len(seq)]

\* All elements of a sequence as a set
Range(seq) == {seq[i] : i \in 1..Len(seq)}

\* Subsequence from index i to end
Suffix(seq, i) == SubSeq(seq, i, Len(seq))

\* Prefix of length n
Prefix(seq, n) == SubSeq(seq, 1, n)

\* Whether seq1 is a prefix of seq2
IsPrefix(seq1, seq2) ==
    /\ Len(seq1) <= Len(seq2)
    /\ \A i \in 1..Len(seq1) : seq1[i] = seq2[i]

\* ========================================================================
\* Log Entry Operations (used by Raft spec)
\* ========================================================================

\* A log entry record: [term |-> Nat, index |-> Nat, value |-> Value]
\* These are abstract -- the Raft spec instantiates them with block data.

\* Whether log entry e1 is at least as up-to-date as e2
\* (used for vote granting: Section 5.4.1 of the Raft paper)
LogUpToDate(lastTerm1, lastIndex1, lastTerm2, lastIndex2) ==
    \/ lastTerm1 > lastTerm2
    \/ (lastTerm1 = lastTerm2 /\ lastIndex1 >= lastIndex2)

\* ========================================================================
\* Bounded Model Checking Helpers
\* ========================================================================

\* Useful for bounding state space in TLC model checking
BoundedNat(n) == 0..n
BoundedPosNat(n) == 1..n

====
