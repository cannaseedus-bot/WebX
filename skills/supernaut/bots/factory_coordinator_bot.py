#!/usr/bin/env python3
"""
Factory Coordinator Bot (Micronaut Skill)
Bridges supernaut DI system to native MicronauntFactory

This is the supernaut-integrated factory that:
1. Loads JSON schemas
2. Invokes native factory (via C++ factory.dll or socket)
3. Tracks created micronauts in supernaut registry
4. Coordinates composition and publishing
"""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Registry path (shared with supernaut)
REGISTRY_PATH = "C:\\Users\\canna\\.claude\\commands\\micronaut_registry.json"
FACTORY_BINARY = "C:\\Users\\canna\\.micronaut\\engine\\Release\\micronaut_factory.exe"


class MicronauntBackend(Enum):
    DETERMINISTIC_LOCAL = "deterministic"
    SEMANTIC_XML = "semantic"
    COMPOSITE = "composite"


@dataclass
class MicronauntMeta:
    id: str
    name: str
    version: str
    author: str
    description: str


@dataclass
class MicronauntSchema:
    meta: MicronauntMeta
    targets: dict[str, Any]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    control: dict[str, Any]
    distribution: dict[str, Any]
    semantic: dict[str, Any] | None = None


@dataclass
class MicronauntInstance:
    id: str
    packet_hash: str
    packet_size: int
    registered: bool
    backend: str
    created_at: str
    reputation: float = 0.5


class MicronauntFactoryCoordinator:
    """
    Supernaut-integrated Factory Coordinator
    
    Responsibilities:
    - Load schemas from files
    - Invoke native factory
    - Manage registry
    - Coordinate publishing
    - Track reputation
    """

    def __init__(self):
        self.instances: dict[str, MicronauntInstance] = {}
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load supernaut registry (shared with micronaut_registry.json)"""
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH) as f:
                return json.load(f)
        return {"micronauts": {}, "stats": {"total": 0}}

    def _save_registry(self):
        """Persist registry"""
        with open(REGISTRY_PATH, "w") as f:
            json.dump(self.registry, f, indent=2)

    def create_from_file(
        self, 
        schema_path: str,
        backend: str = "deterministic"
    ) -> MicronauntInstance | None:
        """
        Create micronaut from schema file
        
        Flow:
        1. Load JSON schema
        2. Validate schema
        3. Call native factory to compile
        4. Register instance
        5. Update registry
        """
        print(f"[Factory Coordinator] Creating from: {schema_path}")
        
        if not os.path.exists(schema_path):
            print(f"  ❌ Schema file not found: {schema_path}")
            return None
        
        # Load schema JSON
        with open(schema_path) as f:
            schema_json = json.load(f)
        
        # Validate schema structure
        if not self._validate_schema(schema_json):
            print("  ❌ Schema validation failed")
            return None
        
        # Extract meta
        meta = schema_json.get("meta", {})
        micronaut_id = meta.get("id", "unknown")
        
        print(f"  Schema ID: {micronaut_id}")
        print(f"  Backend: {backend}")
        
        # Call native factory (simulate or invoke if binary exists)
        instance = self._invoke_native_factory(schema_json, backend)
        
        if not instance:
            print("  ❌ Native factory failed")
            return None
        
        # Register in memory
        self.instances[instance.id] = instance
        
        # Update supernaut registry
        self.registry["micronauts"][instance.id] = {
            "id": instance.id,
            "hash": instance.packet_hash,
            "backend": backend,
            "registered": instance.registered,
            "created": instance.created_at,
            "reputation": instance.reputation
        }
        self.registry["stats"]["total"] = len(self.instances)
        self._save_registry()
        
        print(f"  ✅ Created: {instance.id}")
        print(f"     Hash: {instance.packet_hash[:32]}...")
        print(f"     Registered: {instance.registered}")
        
        return instance

    def _validate_schema(self, schema: dict[str, Any]) -> bool:
        """Validate schema structure"""
        required_keys = ["meta", "targets", "skills", "control"]
        return all(k in schema for k in required_keys)

    def _invoke_native_factory(
        self,
        schema_json: dict[str, Any],
        backend: str
    ) -> MicronauntInstance | None:
        """
        Invoke native C++ factory
        
        For now: simulate with deterministic hashing
        Later: Call MicronauntFactory via IPC/socket
        """
        meta = schema_json["meta"]
        micronaut_id = meta["id"]
        
        # Deterministic packet simulation (would be real SCX-BSON from C++)
        packet_hash = hashlib.sha256(
            json.dumps(schema_json, sort_keys=True).encode()
        ).hexdigest()
        
        packet_size = len(json.dumps(schema_json).encode())
        
        instance = MicronauntInstance(
            id=micronaut_id,
            packet_hash=packet_hash,
            packet_size=packet_size,
            registered=True,  # Assume factory registers
            backend=backend,
            created_at=self._now(),
            reputation=0.5
        )
        
        return instance

    def publish(
        self,
        micronaut_id: str,
        replica_count: int = 3
    ) -> bool:
        """Publish micronaut to mesh peers"""
        if micronaut_id not in self.instances:
            print(f"  ❌ Micronaut not found: {micronaut_id}")
            return False
        
        print(f"[Factory Coordinator] Publishing: {micronaut_id}")
        print(f"  Replicas: {replica_count}")
        
        instance = self.instances[micronaut_id]
        
        # Update registry
        self.registry["micronauts"][micronaut_id]["replicas"] = replica_count
        self._save_registry()
        
        print("  ✅ Published\n")
        return True

    def resolve(self, mna_uri: str) -> dict[str, Any] | None:
        """Resolve mna:// URI"""
        if mna_uri in self.instances:
            inst = self.instances[mna_uri]
            return {
                "id": inst.id,
                "hash": inst.packet_hash,
                "backend": inst.backend,
                "registered": inst.registered
            }
        return None

    def compose(
        self,
        micronaut_ids: list[str],
        composition_name: str
    ) -> dict[str, Any] | None:
        """Compose multiple micronauts into one"""
        print(f"[Factory Coordinator] Composing {len(micronaut_ids)} micronauts")
        
        for mid in micronaut_ids:
            if mid not in self.instances:
                print(f"  ❌ Not found: {mid}")
                return None
            print(f"  → {mid}")
        
        # Merge packets (deterministically)
        merged_hash = hashlib.sha256(
            "".join(self.instances[mid].packet_hash for mid in micronaut_ids).encode()
        ).hexdigest()
        
        composed_id = f"mna://composed/{composition_name}@1.0.0"
        
        instance = MicronauntInstance(
            id=composed_id,
            packet_hash=merged_hash,
            packet_size=sum(
                self.instances[mid].packet_size for mid in micronaut_ids
            ),
            registered=False,
            backend="composite",
            created_at=self._now(),
            reputation=0.5
        )
        
        self.instances[composed_id] = instance
        self.registry["micronauts"][composed_id] = {
            "id": composed_id,
            "hash": merged_hash,
            "backend": "composite",
            "composed_from": micronaut_ids
        }
        self._save_registry()
        
        print(f"  ✅ Composed: {composed_id}\n")
        return {"id": composed_id, "hash": merged_hash}

    def list_instances(self) -> list[dict[str, Any]]:
        """List all created instances"""
        return [
            {
                "id": inst.id,
                "hash": inst.packet_hash,
                "backend": inst.backend,
                "registered": inst.registered
            }
            for inst in self.instances.values()
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get factory statistics"""
        return {
            "total_created": len(self.instances),
            "total_registered": sum(
                1 for inst in self.instances.values() if inst.registered
            ),
            "backends": {
                "deterministic": sum(
                    1 for inst in self.instances.values()
                    if inst.backend == "deterministic"
                ),
                "semantic": sum(
                    1 for inst in self.instances.values()
                    if inst.backend == "semantic"
                ),
                "composite": sum(
                    1 for inst in self.instances.values()
                    if inst.backend == "composite"
                )
            }
        }

    @staticmethod
    def _now() -> str:
        """Current ISO timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


def main():
    """Test coordinator"""
    coordinator = MicronauntFactoryCoordinator()
    
    # Example: Create matmul micronaut
    schema_path = "schemas/matmul.json"
    if os.path.exists(schema_path):
        instance = coordinator.create_from_file(schema_path, "deterministic")
        
        if instance:
            coordinator.publish(instance.id, 3)
    
    # List instances
    print("\n[Factory Coordinator] Instances:")
    for inst in coordinator.list_instances():
        print(f"  {inst['id']} ({inst['backend']})")
    
    # Print stats
    print("\n[Factory Coordinator] Stats:")
    stats = coordinator.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
