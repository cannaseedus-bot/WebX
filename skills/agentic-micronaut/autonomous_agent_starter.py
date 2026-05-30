# Python 3 - Agentic Micronaut Autonomous Agent Starter
# Required baseline for autonomous-agent tasks

import random
from typing import List, Dict, Any, Optional
import json


class Agent:
    """Autonomous agent with goals, observations, and learning"""
    
    def __init__(self, name: str, goals: List[str], agent_type: str = "generic"):
        self.name = name
        self.goals = goals
        self.agent_type = agent_type
        self.knowledge = {}
        self.state = "idle"
        self.tick = 0
    
    def sense(self, environment: List[str]) -> str:
        """Observe environment and return single observation"""
        observation = random.choice(environment)
        print(f"[{self.name}] Observed: {observation}")
        return observation
    
    def decide(self, observation: str) -> str:
        """Decide on action based on observation and goals"""
        for goal in self.goals:
            if goal.lower() in observation.lower():
                print(f"[{self.name}] Deciding to act towards goal: {goal}")
                return goal
        return "explore"
    
    def act(self, action: str) -> Dict[str, Any]:
        """Execute action and return result"""
        if action == "explore":
            print(f"[{self.name}] Exploring environment...")
            self.state = "exploring"
        else:
            print(f"[{self.name}] Working on goal: {action}")
            self.state = "working"
        
        return {
            "ok": True,
            "action": action,
            "agent": self.name,
            "tick": self.tick
        }
    
    def learn(self, observation: str, action: str, result: Dict[str, Any]) -> None:
        """Learn from experience"""
        key = f"{observation}_{action}"
        self.knowledge[key] = result
        print(f"[{self.name}] Knowledge updated: learned {len(self.knowledge)} patterns")
    
    def step(self, environment: List[str]) -> Dict[str, Any]:
        """Execute single agent step: sense -> decide -> act -> learn"""
        self.tick += 1
        observation = self.sense(environment)
        action = self.decide(observation)
        result = self.act(action)
        self.learn(observation, action, result)
        return result
    
    def run(self, environment: List[str], steps: int = 5) -> List[Dict[str, Any]]:
        """Run agent for N steps"""
        results = []
        for _ in range(steps):
            print(f"\n--- Agent Step {self.tick + 1} ---")
            result = self.step(environment)
            results.append(result)
        return results


class MultiAgentSystem:
    """Coordinate multiple agents with shared environment"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.environment = []
        self.history = []
    
    def add_observation(self, observation: str) -> None:
        """Add observation to shared environment"""
        self.environment.append(observation)
    
    def step(self) -> Dict[str, Any]:
        """Execute one step for all agents"""
        results = []
        for agent in self.agents:
            result = agent.step(self.environment)
            results.append(result)
        
        step_result = {
            "agents_acted": len(self.agents),
            "results": results,
            "environment_size": len(self.environment)
        }
        self.history.append(step_result)
        return step_result
    
    def run(self, steps: int = 5) -> List[Dict[str, Any]]:
        """Run all agents for N steps"""
        for _ in range(steps):
            self.step()
        return self.history


# Example: Single autonomous agent
if __name__ == "__main__":
    # Single agent example
    environment = [
        "New email about project deadline",
        "Low battery warning",
        "Weather is sunny",
        "Task: write report",
        "Meeting reminder",
    ]
    
    goals = ["write report", "charge battery"]
    agent = Agent("AutonomousAgent", goals, agent_type="executor")
    
    print("=== Single Agent Run ===")
    results = agent.run(environment, steps=5)
    
    # Multi-agent example
    print("\n\n=== Multi-Agent System ===")
    planner = Agent("Planner", ["create plan", "decompose task"], agent_type="planner")
    executor = Agent("Executor", ["execute task", "report progress"], agent_type="executor")
    coordinator = Agent("Coordinator", ["coordinate", "resolve conflict"], agent_type="coordinator")
    
    system = MultiAgentSystem([planner, executor, coordinator])
    
    # Add observations
    for obs in environment:
        system.add_observation(obs)
    
    # Run system
    history = system.run(steps=3)
    
    print("\n=== System History ===")
    print(json.dumps(history, indent=2))
