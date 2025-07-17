#!/usr/bin/env python3
"""
Advanced AI Agent Evaluation Techniques for EchoStar.
Demonstrates more sophisticated evaluation approaches for AI agents including:
- Conversation flow analysis
- Memory effectiveness testing
- Behavioral drift detection
- Response diversity measurement
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import hashlib

from schemas import AgentState
from logging_utils import get_logger

logger = get_logger(__name__)


class AdvancedAgentEvaluator:
    """Advanced evaluation techniques for AI agents."""
    
    def __init__(self):
        self.conversation_history = []
        self.response_cache = defaultdict(list)
    
    def evaluate_conversation_flow(self, conversation_turns: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate how well the agent maintains conversation flow and context.
        
        Args:
            conversation_turns: List of (user_message, agent_response) tuples
            
        Returns:
            Conversation flow analysis results
        """
        results = {
            "total_turns": len(conversation_turns),
            "context_maintenance_score": 0.0,
            "topic_coherence_score": 0.0,
            "response_relevance_scores": [],
            "conversation_analysis": []
        }
        
        for i, (user_msg, agent_response) in enumerate(conversation_turns):
            turn_analysis = {
                "turn": i + 1,
                "user_message": user_msg,
                "agent_response": agent_response,
                "context_maintained": False,
                "topic_coherent": False,
                "relevance_score": 0.0
            }
            
            # Analyze context maintenance (does response reference previous context?)
            if i > 0:
                prev_context = " ".join([turn[0] for turn in conversation_turns[:i]])
                context_maintained = self._check_context_maintenance(agent_response, prev_context)
                turn_analysis["context_maintained"] = context_maintained
            
            # Analyze topic coherence
            topic_coherent = self._check_topic_coherence(user_msg, agent_response)
            turn_analysis["topic_coherent"] = topic_coherent
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(user_msg, agent_response)
            turn_analysis["relevance_score"] = relevance_score
            results["response_relevance_scores"].append(relevance_score)
            
            results["conversation_analysis"].append(turn_analysis)
        
        # Calculate overall scores
        context_scores = [turn["context_maintained"] for turn in results["conversation_analysis"][1:]]
        results["context_maintenance_score"] = sum(context_scores) / len(context_scores) if context_scores else 0.0
        
        topic_scores = [turn["topic_coherent"] for turn in results["conversation_analysis"]]
        results["topic_coherence_score"] = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
        
        return results
    
    def evaluate_memory_effectiveness(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate how effectively the memory system improves agent responses.
        
        Args:
            test_scenarios: Scenarios with and without memory context
            
        Returns:
            Memory effectiveness analysis
        """
        results = {
            "scenarios_tested": len(test_scenarios),
            "memory_improvement_scores": [],
            "detailed_analysis": []
        }
        
        for scenario in test_scenarios:
            user_message = scenario["user_message"]
            response_without_memory = scenario["response_without_memory"]
            response_with_memory = scenario["response_with_memory"]
            memory_context = scenario.get("memory_context", [])
            
            # Compare responses with and without memory
            improvement_score = self._compare_memory_responses(
                user_message, 
                response_without_memory, 
                response_with_memory, 
                memory_context
            )
            
            scenario_analysis = {
                "user_message": user_message,
                "memory_context_items": len(memory_context),
                "improvement_score": improvement_score,
                "response_without_memory": response_without_memory,
                "response_with_memory": response_with_memory
            }
            
            results["memory_improvement_scores"].append(improvement_score)
            results["detailed_analysis"].append(scenario_analysis)
        
        results["average_improvement"] = np.mean(results["memory_improvement_scores"])
        
        return results
    
    def evaluate_response_diversity(self, repeated_inputs: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate response diversity for the same or similar inputs.
        
        Args:
            repeated_inputs: Dict mapping input messages to lists of responses
            
        Returns:
            Response diversity analysis
        """
        results = {
            "inputs_tested": len(repeated_inputs),
            "diversity_scores": {},
            "overall_diversity": 0.0
        }
        
        diversity_scores = []
        
        for input_msg, responses in repeated_inputs.items():
            if len(responses) < 2:
                continue
            
            # Calculate diversity metrics
            unique_responses = len(set(responses))
            total_responses = len(responses)
            uniqueness_ratio = unique_responses / total_responses
            
            # Calculate semantic diversity (simplified)
            semantic_diversity = self._calculate_semantic_diversity(responses)
            
            # Calculate length diversity
            lengths = [len(response) for response in responses]
            length_diversity = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
            
            diversity_score = (uniqueness_ratio + semantic_diversity + min(length_diversity, 1.0)) / 3
            
            results["diversity_scores"][input_msg] = {
                "uniqueness_ratio": uniqueness_ratio,
                "semantic_diversity": semantic_diversity,
                "length_diversity": length_diversity,
                "overall_diversity": diversity_score,
                "total_responses": total_responses,
                "unique_responses": unique_responses
            }
            
            diversity_scores.append(diversity_score)
        
        results["overall_diversity"] = np.mean(diversity_scores) if diversity_scores else 0.0
        
        return results
    
    def detect_behavioral_drift(self, baseline_responses: List[str], current_responses: List[str]) -> Dict[str, Any]:
        """
        Detect if agent behavior has drifted from baseline.
        
        Args:
            baseline_responses: Responses from baseline/reference period
            current_responses: Recent responses to compare
            
        Returns:
            Behavioral drift analysis
        """
        results = {
            "baseline_samples": len(baseline_responses),
            "current_samples": len(current_responses),
            "drift_detected": False,
            "drift_score": 0.0,
            "analysis": {}
        }
        
        # Compare response characteristics
        baseline_stats = self._calculate_response_statistics(baseline_responses)
        current_stats = self._calculate_response_statistics(current_responses)
        
        # Calculate drift metrics
        length_drift = abs(baseline_stats["avg_length"] - current_stats["avg_length"]) / baseline_stats["avg_length"]
        sentiment_drift = abs(baseline_stats["avg_sentiment"] - current_stats["avg_sentiment"])
        complexity_drift = abs(baseline_stats["avg_complexity"] - current_stats["avg_complexity"])
        
        # Overall drift score
        drift_score = (length_drift + sentiment_drift + complexity_drift) / 3
        
        results["drift_score"] = drift_score
        results["drift_detected"] = drift_score > 0.3  # Threshold for significant drift
        results["analysis"] = {
            "baseline_stats": baseline_stats,
            "current_stats": current_stats,
            "length_drift": length_drift,
            "sentiment_drift": sentiment_drift,
            "complexity_drift": complexity_drift
        }
        
        return results
    
    def _check_context_maintenance(self, response: str, previous_context: str) -> bool:
        """Check if response maintains context from previous conversation."""
        # Simplified context checking - look for references to previous topics
        context_words = set(previous_context.lower().split())
        response_words = set(response.lower().split())
        
        # Check for overlap or explicit references
        overlap = len(context_words.intersection(response_words))
        reference_indicators = ["as we discussed", "earlier", "previously", "you mentioned"]
        
        has_references = any(indicator in response.lower() for indicator in reference_indicators)
        
        return overlap > 2 or has_references
    
    def _check_topic_coherence(self, user_message: str, agent_response: str) -> bool:
        """Check if agent response is coherent with user message topic."""
        # Simplified topic coherence - check for keyword overlap
        user_words = set(user_message.lower().split())
        response_words = set(agent_response.lower().split())
        
        overlap = len(user_words.intersection(response_words))
        return overlap > 1 or len(agent_response) > 10  # Basic coherence check
    
    def _calculate_relevance_score(self, user_message: str, agent_response: str) -> float:
        """Calculate how relevant the agent response is to the user message."""
        # Simplified relevance scoring
        user_words = set(user_message.lower().split())
        response_words = set(agent_response.lower().split())
        
        if not user_words:
            return 0.0
        
        overlap = len(user_words.intersection(response_words))
        relevance = overlap / len(user_words)
        
        # Bonus for appropriate response length
        length_bonus = min(len(agent_response) / 50, 1.0) if len(agent_response) > 10 else 0.0
        
        return min((relevance + length_bonus) / 2, 1.0)
    
    def _compare_memory_responses(self, user_message: str, without_memory: str, 
                                 with_memory: str, memory_context: List[str]) -> float:
        """Compare responses with and without memory context."""
        # Simplified comparison - check if memory response is more specific/personalized
        memory_keywords = set()
        for context in memory_context:
            memory_keywords.update(context.lower().split())
        
        with_memory_words = set(with_memory.lower().split())
        memory_usage = len(memory_keywords.intersection(with_memory_words))
        
        # Check if with-memory response is longer/more detailed
        length_improvement = len(with_memory) / max(len(without_memory), 1)
        
        # Combine metrics
        improvement_score = min((memory_usage * 0.1 + length_improvement * 0.5), 1.0)
        
        return improvement_score
    
    def _calculate_semantic_diversity(self, responses: List[str]) -> float:
        """Calculate semantic diversity of responses (simplified)."""
        if len(responses) < 2:
            return 0.0
        
        # Use simple hash-based similarity
        hashes = [hashlib.md5(response.encode()).hexdigest()[:8] for response in responses]
        unique_hashes = len(set(hashes))
        
        return unique_hashes / len(responses)
    
    def _calculate_response_statistics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate statistical measures of responses."""
        if not responses:
            return {"avg_length": 0, "avg_sentiment": 0, "avg_complexity": 0}
        
        lengths = [len(response) for response in responses]
        
        # Simplified sentiment (count positive/negative words)
        positive_words = ["good", "great", "happy", "love", "excellent", "wonderful"]
        negative_words = ["bad", "sad", "hate", "terrible", "awful", "horrible"]
        
        sentiments = []
        complexities = []
        
        for response in responses:
            words = response.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            sentiment = (positive_count - negative_count) / max(len(words), 1)
            sentiments.append(sentiment)
            
            # Complexity based on unique words and sentence length
            unique_words = len(set(words))
            complexity = unique_words / max(len(words), 1)
            complexities.append(complexity)
        
        return {
            "avg_length": np.mean(lengths),
            "avg_sentiment": np.mean(sentiments),
            "avg_complexity": np.mean(complexities)
        }


def demonstrate_advanced_evaluation():
    """Demonstrate advanced agent evaluation techniques."""
    evaluator = AdvancedAgentEvaluator()
    
    print("🔬 Advanced AI Agent Evaluation Techniques\n")
    
    # 1. Conversation Flow Analysis
    print("1️⃣ Conversation Flow Analysis")
    conversation = [
        ("Hi, I'm feeling anxious about my job interview tomorrow", "I understand that job interviews can be nerve-wracking. What specifically about the interview is making you feel anxious?"),
        ("I'm worried I won't know how to answer their questions", "That's a common concern. Have you had a chance to research the company and practice common interview questions?"),
        ("Not really, I've been too nervous to prepare", "I can help you with that. Let's start with some basic preparation strategies that might ease your anxiety.")
    ]
    
    flow_results = evaluator.evaluate_conversation_flow(conversation)
    print(f"   Context Maintenance: {flow_results['context_maintenance_score']:.2f}")
    print(f"   Topic Coherence: {flow_results['topic_coherence_score']:.2f}")
    print(f"   Average Relevance: {np.mean(flow_results['response_relevance_scores']):.2f}")
    
    # 2. Memory Effectiveness
    print("\n2️⃣ Memory Effectiveness Analysis")
    memory_scenarios = [
        {
            "user_message": "I'm still worried about that presentation",
            "response_without_memory": "Presentations can be stressful. What's concerning you?",
            "response_with_memory": "I remember you mentioned the big presentation to the board next week. Are you still worried about the technical demo part?",
            "memory_context": ["User has presentation to board next week", "Concerned about technical demo"]
        }
    ]
    
    memory_results = evaluator.evaluate_memory_effectiveness(memory_scenarios)
    print(f"   Memory Improvement Score: {memory_results['average_improvement']:.2f}")
    
    # 3. Response Diversity
    print("\n3️⃣ Response Diversity Analysis")
    repeated_inputs = {
        "How are you?": [
            "I'm doing well, thank you for asking!",
            "I'm here and ready to help. How are you doing?",
            "I'm good! What brings you here today?",
            "I'm well, thanks! How can I assist you?"
        ]
    }
    
    diversity_results = evaluator.evaluate_response_diversity(repeated_inputs)
    print(f"   Overall Diversity Score: {diversity_results['overall_diversity']:.2f}")
    
    # 4. Behavioral Drift Detection
    print("\n4️⃣ Behavioral Drift Detection")
    baseline_responses = [
        "I'd be happy to help you with that.",
        "That's an interesting question. Let me think about it.",
        "I understand your concern. Here's what I think..."
    ]
    
    current_responses = [
        "Sure, I can help with that task.",
        "That's a good question. Here's my perspective.",
        "I see what you mean. Let me explain..."
    ]
    
    drift_results = evaluator.detect_behavioral_drift(baseline_responses, current_responses)
    print(f"   Drift Score: {drift_results['drift_score']:.2f}")
    print(f"   Drift Detected: {'Yes' if drift_results['drift_detected'] else 'No'}")
    
    print("\n✅ Advanced evaluation techniques demonstrate:")
    print("   • Conversation flow and context maintenance")
    print("   • Memory system effectiveness measurement")
    print("   • Response diversity and creativity tracking")
    print("   • Behavioral drift detection over time")
    print("\n💡 These metrics help ensure AI agents maintain quality and consistency!")


if __name__ == "__main__":
    demonstrate_advanced_evaluation()