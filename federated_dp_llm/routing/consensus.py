"""
Consensus Management for Federated Inference

Implements consensus algorithms for ensuring agreement across multiple
federated nodes in critical healthcare inference scenarios.
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
from collections import Counter, defaultdict


class ConsensusAlgorithm(Enum):
    """Supported consensus algorithms."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BYZANTINE_FAULT_TOLERANT = "bft"
    PROOF_OF_AUTHORITY = "poa"


@dataclass
class ConsensusNode:
    """Represents a node participating in consensus."""
    node_id: str
    weight: float = 1.0
    reputation: float = 1.0
    last_seen: float = field(default_factory=time.time)
    is_byzantine: bool = False  # For testing fault tolerance
    response_history: List[bool] = field(default_factory=list)


@dataclass
class ConsensusProposal:
    """A proposal for consensus decision."""
    proposal_id: str
    proposer_id: str
    content: str
    confidence_score: float
    timestamp: float
    signatures: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConsensusRound:
    """Represents a single consensus round."""
    round_id: str
    proposals: List[ConsensusProposal]
    votes: Dict[str, str]  # node_id -> proposal_id
    participants: Set[str]
    required_nodes: int
    algorithm: ConsensusAlgorithm
    status: str = "active"  # active, completed, failed
    start_time: float = field(default_factory=time.time)
    timeout: float = 30.0


class ByzantineFaultTolerant:
    """Byzantine Fault Tolerant consensus implementation."""
    
    def __init__(self, max_byzantine_nodes: int):
        self.max_byzantine_nodes = max_byzantine_nodes
        self.min_honest_nodes = 2 * max_byzantine_nodes + 1
    
    def can_achieve_consensus(self, total_nodes: int) -> bool:
        """Check if consensus is possible with given node count."""
        return total_nodes >= self.min_honest_nodes
    
    def validate_proposal(self, proposal: ConsensusProposal, node_reputation: float) -> bool:
        """Validate a proposal considering node reputation."""
        # Simple validation based on node reputation and proposal structure
        if node_reputation < 0.5:  # Low reputation nodes
            return False
        
        if proposal.confidence_score < 0.3:  # Low confidence proposals
            return False
        
        # Check proposal content validity (simplified)
        if len(proposal.content.strip()) < 10:
            return False
        
        return True
    
    def detect_byzantine_behavior(self, node_id: str, node: ConsensusNode) -> bool:
        """Detect potential Byzantine behavior patterns."""
        if len(node.response_history) < 5:
            return False
        
        # Check for inconsistent voting patterns
        recent_responses = node.response_history[-10:]
        success_rate = sum(recent_responses) / len(recent_responses)
        
        # Flag nodes with very low success rate as potentially Byzantine
        return success_rate < 0.3
    
    def calculate_byzantine_resilient_threshold(self, total_nodes: int) -> int:
        """Calculate threshold for Byzantine-resilient consensus."""
        return (2 * total_nodes // 3) + 1


class ConsensusManager:
    """Main consensus management system."""
    
    def __init__(
        self,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
        max_byzantine_nodes: int = 1
    ):
        self.algorithm = algorithm
        self.nodes: Dict[str, ConsensusNode] = {}
        self.active_rounds: Dict[str, ConsensusRound] = {}
        self.consensus_history: List[Tuple[str, str, bool, float]] = []  # round_id, result, success, timestamp
        
        # Byzantine fault tolerance
        self.bft = ByzantineFaultTolerant(max_byzantine_nodes)
        
        # Reputation tracking
        self.reputation_decay = 0.95  # Decay factor for reputation
        self.reputation_update_interval = 3600  # 1 hour
        self.last_reputation_update = time.time()
    
    def register_node(self, node_id: str, weight: float = 1.0, reputation: float = 1.0):
        """Register a node for consensus participation."""
        node = ConsensusNode(
            node_id=node_id,
            weight=weight,
            reputation=reputation
        )
        self.nodes[node_id] = node
    
    def update_node_reputation(self, node_id: str, success: bool):
        """Update node reputation based on performance."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Update response history
        node.response_history.append(success)
        if len(node.response_history) > 100:  # Keep last 100 responses
            node.response_history = node.response_history[-100:]
        
        # Calculate reputation based on recent performance
        if len(node.response_history) >= 5:
            recent_success_rate = sum(node.response_history[-20:]) / min(20, len(node.response_history))
            # Exponential moving average for reputation
            node.reputation = 0.7 * node.reputation + 0.3 * recent_success_rate
        
        # Detect Byzantine behavior
        if self.bft.detect_byzantine_behavior(node_id, node):
            node.is_byzantine = True
            node.weight = 0.1  # Reduce weight for suspected Byzantine nodes
    
    async def start_consensus_round(
        self,
        round_id: str,
        proposals: List[ConsensusProposal],
        participant_ids: List[str],
        timeout: float = 30.0
    ) -> ConsensusRound:
        """Start a new consensus round."""
        
        # Validate participants
        valid_participants = {
            node_id for node_id in participant_ids
            if node_id in self.nodes and not self.nodes[node_id].is_byzantine
        }
        
        if len(valid_participants) < 3:
            raise ValueError("Insufficient valid participants for consensus")
        
        # Check Byzantine fault tolerance
        if self.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            if not self.bft.can_achieve_consensus(len(valid_participants)):
                raise ValueError("Insufficient nodes for Byzantine fault tolerant consensus")
        
        # Determine required nodes based on algorithm
        if self.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            required_nodes = self.bft.calculate_byzantine_resilient_threshold(len(valid_participants))
        else:
            required_nodes = len(valid_participants) // 2 + 1  # Simple majority
        
        # Create consensus round
        consensus_round = ConsensusRound(
            round_id=round_id,
            proposals=proposals,
            votes={},
            participants=valid_participants,
            required_nodes=required_nodes,
            algorithm=self.algorithm,
            timeout=timeout
        )
        
        self.active_rounds[round_id] = consensus_round
        
        # Start timeout handler
        asyncio.create_task(self._handle_round_timeout(round_id, timeout))
        
        return consensus_round
    
    async def submit_vote(self, round_id: str, node_id: str, proposal_id: str, signature: str = None) -> bool:
        """Submit a vote for a proposal in a consensus round."""
        if round_id not in self.active_rounds:
            raise ValueError(f"Consensus round {round_id} not found")
        
        consensus_round = self.active_rounds[round_id]
        
        if consensus_round.status != "active":
            raise ValueError(f"Consensus round {round_id} is not active")
        
        if node_id not in consensus_round.participants:
            raise ValueError(f"Node {node_id} not authorized for this consensus round")
        
        if node_id in consensus_round.votes:
            raise ValueError(f"Node {node_id} has already voted")
        
        # Validate proposal exists
        proposal_ids = {p.proposal_id for p in consensus_round.proposals}
        if proposal_id not in proposal_ids:
            raise ValueError(f"Proposal {proposal_id} not found in round")
        
        # Record vote
        consensus_round.votes[node_id] = proposal_id
        
        # Add signature if provided
        if signature:
            for proposal in consensus_round.proposals:
                if proposal.proposal_id == proposal_id:
                    proposal.signatures[node_id] = signature
                    break
        
        # Check if consensus achieved
        if len(consensus_round.votes) >= consensus_round.required_nodes:
            await self._evaluate_consensus(round_id)
        
        return True
    
    async def _evaluate_consensus(self, round_id: str) -> Optional[str]:
        """Evaluate whether consensus has been achieved."""
        consensus_round = self.active_rounds[round_id]
        
        if self.algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            result = self._majority_vote_consensus(consensus_round)
        elif self.algorithm == ConsensusAlgorithm.WEIGHTED_VOTE:
            result = self._weighted_vote_consensus(consensus_round)
        elif self.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            result = self._bft_consensus(consensus_round)
        elif self.algorithm == ConsensusAlgorithm.PROOF_OF_AUTHORITY:
            result = self._proof_of_authority_consensus(consensus_round)
        else:
            result = None
        
        if result:
            consensus_round.status = "completed"
            self._record_consensus_result(round_id, result, True)
            
            # Update node reputations
            for node_id in consensus_round.votes:
                voted_for_winner = consensus_round.votes[node_id] == result
                self.update_node_reputation(node_id, voted_for_winner)
        
        return result
    
    def _majority_vote_consensus(self, consensus_round: ConsensusRound) -> Optional[str]:
        """Simple majority vote consensus."""
        vote_counts = Counter(consensus_round.votes.values())
        
        if not vote_counts:
            return None
        
        most_voted, count = vote_counts.most_common(1)[0]
        
        # Check if majority achieved
        if count > len(consensus_round.participants) // 2:
            return most_voted
        
        return None
    
    def _weighted_vote_consensus(self, consensus_round: ConsensusRound) -> Optional[str]:
        """Weighted vote consensus considering node weights and reputation."""
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for node_id, proposal_id in consensus_round.votes.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                weight = node.weight * node.reputation
                weighted_votes[proposal_id] += weight
                total_weight += weight
        
        if not weighted_votes:
            return None
        
        # Find proposal with highest weighted score
        best_proposal = max(weighted_votes.items(), key=lambda x: x[1])
        proposal_id, weight = best_proposal
        
        # Check if weighted majority achieved (>50% of total weight)
        if weight > total_weight / 2:
            return proposal_id
        
        return None
    
    def _bft_consensus(self, consensus_round: ConsensusRound) -> Optional[str]:
        """Byzantine Fault Tolerant consensus."""
        # Only consider votes from non-Byzantine nodes
        honest_votes = {
            node_id: proposal_id
            for node_id, proposal_id in consensus_round.votes.items()
            if node_id in self.nodes and not self.nodes[node_id].is_byzantine
        }
        
        vote_counts = Counter(honest_votes.values())
        
        if not vote_counts:
            return None
        
        most_voted, count = vote_counts.most_common(1)[0]
        
        # Byzantine fault tolerant threshold
        required_votes = self.bft.calculate_byzantine_resilient_threshold(len(honest_votes))
        
        if count >= required_votes:
            return most_voted
        
        return None
    
    def _proof_of_authority_consensus(self, consensus_round: ConsensusRound) -> Optional[str]:
        """Proof of Authority consensus (authority nodes have higher weight)."""
        # Identify authority nodes (highest reputation nodes)
        authority_nodes = sorted(
            [node_id for node_id in consensus_round.participants if node_id in self.nodes],
            key=lambda nid: self.nodes[nid].reputation,
            reverse=True
        )[:3]  # Top 3 authority nodes
        
        # Count votes from authority nodes
        authority_votes = Counter()
        for node_id in authority_nodes:
            if node_id in consensus_round.votes:
                authority_votes[consensus_round.votes[node_id]] += 1
        
        if not authority_votes:
            # Fall back to regular majority if no authority votes
            return self._majority_vote_consensus(consensus_round)
        
        most_voted, count = authority_votes.most_common(1)[0]
        
        # Need majority of authority nodes
        if count > len(authority_nodes) // 2:
            return most_voted
        
        return None
    
    async def _handle_round_timeout(self, round_id: str, timeout: float):
        """Handle consensus round timeout."""
        await asyncio.sleep(timeout)
        
        if round_id in self.active_rounds:
            consensus_round = self.active_rounds[round_id]
            if consensus_round.status == "active":
                consensus_round.status = "failed"
                self._record_consensus_result(round_id, None, False)
    
    def _record_consensus_result(self, round_id: str, result: Optional[str], success: bool):
        """Record consensus result for analysis."""
        self.consensus_history.append((round_id, result or "failed", success, time.time()))
        
        # Keep last 1000 results
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]
    
    def get_consensus_result(self, round_id: str) -> Optional[str]:
        """Get result of a consensus round."""
        if round_id not in self.active_rounds:
            return None
        
        consensus_round = self.active_rounds[round_id]
        
        if consensus_round.status == "completed":
            return self._evaluate_consensus(round_id)
        
        return None
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get comprehensive consensus statistics."""
        if not self.consensus_history:
            return {"total_rounds": 0}
        
        total_rounds = len(self.consensus_history)
        successful_rounds = sum(1 for _, _, success, _ in self.consensus_history if success)
        
        # Node participation stats
        node_stats = {}
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                "reputation": node.reputation,
                "weight": node.weight,
                "is_byzantine": node.is_byzantine,
                "response_count": len(node.response_history),
                "success_rate": sum(node.response_history) / max(len(node.response_history), 1)
            }
        
        return {
            "total_rounds": total_rounds,
            "success_rate": successful_rounds / total_rounds,
            "active_rounds": len(self.active_rounds),
            "registered_nodes": len(self.nodes),
            "byzantine_nodes": sum(1 for node in self.nodes.values() if node.is_byzantine),
            "algorithm": self.algorithm.value,
            "node_statistics": node_stats,
            "recent_results": self.consensus_history[-10:]  # Last 10 results
        }
    
    def cleanup_completed_rounds(self, max_age: float = 3600):
        """Clean up old completed consensus rounds."""
        current_time = time.time()
        
        rounds_to_remove = []
        for round_id, consensus_round in self.active_rounds.items():
            if (consensus_round.status in ["completed", "failed"] and 
                current_time - consensus_round.start_time > max_age):
                rounds_to_remove.append(round_id)
        
        for round_id in rounds_to_remove:
            del self.active_rounds[round_id]