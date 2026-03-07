TLC = java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar
TLC_OPTS = -deadlock -workers auto -cleanup
TLC_LARGE = java -XX:+UseParallelGC -Xmx8g -jar tla2tools.jar

.PHONY: all ci deep clean raft raft-full raft-prevote tpc membership membership-3node

# CI config: tight bounds, ~10 min total
ci: raft tpc membership

# All configs including deep exploration
all: ci raft-full raft-prevote membership-3node

# Deep configs only (skip CI baseline)
deep: raft-full raft-prevote membership-3node

raft:
	$(TLC) $(TLC_OPTS) -config Raft.cfg Raft.tla

raft-full:
	$(TLC_LARGE) $(TLC_OPTS) -config Raft-full.cfg Raft.tla

raft-prevote:
	$(TLC) $(TLC_OPTS) -config Raft-prevote.cfg Raft.tla

tpc:
	$(TLC) $(TLC_OPTS) -config TwoPhaseCommit.cfg TwoPhaseCommit.tla

membership:
	$(TLC) $(TLC_OPTS) -config Membership.cfg Membership.tla

membership-3node:
	$(TLC) $(TLC_OPTS) -config Membership-3node.cfg Membership.tla

clean:
	rm -rf states/ *.dot
