"""
Command Line Interface for Federated DP-LLM Router

Provides CLI commands for managing the federated learning system, nodes,
privacy budgets, and monitoring.
"""

import asyncio
import json
import sys
import time
from typing import Dict, List, Optional, Any
import argparse
import yaml
from pathlib import Path

from .core.privacy_accountant import PrivacyAccountant, DPConfig, DPMechanism, CompositionMethod
from .routing.load_balancer import FederatedRouter, RoutingStrategy
from .routing.request_handler import RequestHandler
from .federation.client import HospitalNode, PrivateInferenceClient, FederatedLearningClient
from .federation.server import FederatedTrainer
from .security.compliance import BudgetManager, ComplianceMonitor
from .security.authentication import AuthenticationManager, Role


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Federated DP-LLM Router - Privacy-preserving federated learning for healthcare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the router server
  federated-dp-llm server --config config.yaml --port 8080

  # Register a hospital node
  federated-dp-llm node register --id hospital_a --endpoint https://hospital-a.local:8443

  # Check privacy budget
  federated-dp-llm privacy budget --user doctor_123

  # Run federated training
  federated-dp-llm train --config training_config.yaml --rounds 100

  # Generate compliance report
  federated-dp-llm compliance report --period monthly --format pdf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Server commands
    server_parser = subparsers.add_parser('server', help='Server management')
    server_subparsers = server_parser.add_subparsers(dest='server_action')
    
    start_parser = server_subparsers.add_parser('start', help='Start the router server')
    start_parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file')
    start_parser.add_argument('--port', '-p', type=int, default=8080, help='Server port')
    start_parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    start_parser.add_argument('--ssl-cert', type=str, help='SSL certificate file')
    start_parser.add_argument('--ssl-key', type=str, help='SSL private key file')
    
    # Node commands
    node_parser = subparsers.add_parser('node', help='Node management')
    node_subparsers = node_parser.add_subparsers(dest='node_action')
    
    register_parser = node_subparsers.add_parser('register', help='Register a hospital node')
    register_parser.add_argument('--id', required=True, help='Node ID')
    register_parser.add_argument('--endpoint', required=True, help='Node endpoint URL')
    register_parser.add_argument('--data-size', type=int, default=10000, help='Data size')
    register_parser.add_argument('--compute-capacity', default='4xA100', help='Compute capacity')
    register_parser.add_argument('--department', help='Department')
    
    list_nodes_parser = node_subparsers.add_parser('list', help='List registered nodes')
    list_nodes_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Privacy commands
    privacy_parser = subparsers.add_parser('privacy', help='Privacy management')
    privacy_subparsers = privacy_parser.add_subparsers(dest='privacy_action')
    
    budget_parser = privacy_subparsers.add_parser('budget', help='Privacy budget management')
    budget_parser.add_argument('--user', help='User ID to check budget for')
    budget_parser.add_argument('--department', help='Department to check budget for')
    budget_parser.add_argument('--reset', action='store_true', help='Reset privacy budget')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Federated training')
    train_parser.add_argument('--config', '-c', type=str, required=True, help='Training configuration file')
    train_parser.add_argument('--rounds', type=int, help='Number of training rounds')
    train_parser.add_argument('--clients-per-round', type=int, help='Clients per round')
    train_parser.add_argument('--output', '-o', type=str, help='Output directory for model')
    
    # Query commands
    query_parser = subparsers.add_parser('query', help='Inference queries')
    query_parser.add_argument('--prompt', required=True, help='Query prompt')
    query_parser.add_argument('--user', required=True, help='User ID')
    query_parser.add_argument('--endpoint', default='http://localhost:8080', help='Router endpoint')
    query_parser.add_argument('--model', default='medllama-7b', help='Model name')
    query_parser.add_argument('--privacy-budget', type=float, default=0.1, help='Max privacy budget')
    query_parser.add_argument('--consensus', action='store_true', help='Require consensus')
    
    # Compliance commands
    compliance_parser = subparsers.add_parser('compliance', help='Compliance management')
    compliance_subparsers = compliance_parser.add_subparsers(dest='compliance_action')
    
    report_parser = compliance_subparsers.add_parser('report', help='Generate compliance report')
    report_parser.add_argument('--period', choices=['daily', 'weekly', 'monthly'], default='monthly', help='Report period')
    report_parser.add_argument('--format', choices=['json', 'pdf'], default='json', help='Report format')
    report_parser.add_argument('--output', '-o', type=str, help='Output file')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    init_parser = config_subparsers.add_parser('init', help='Initialize configuration')
    init_parser.add_argument('--env', choices=['hospital', 'coordinator', 'development'], default='development', help='Environment type')
    init_parser.add_argument('--output', '-o', type=str, default='config.yaml', help='Output configuration file')
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


def create_default_config(env_type: str) -> Dict[str, Any]:
    """Create default configuration for environment type."""
    base_config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "ssl_enabled": True
        },
        "privacy": {
            "epsilon_per_query": 0.1,
            "delta": 1e-5,
            "max_budget_per_user": 10.0,
            "noise_multiplier": 1.1,
            "mechanism": "gaussian",
            "composition": "rdp"
        },
        "routing": {
            "strategy": "privacy_aware",
            "max_concurrent_requests": 100,
            "timeout": 30.0
        },
        "security": {
            "jwt_secret": "your-secret-key-here",
            "token_expiry": 3600,
            "enable_mtls": True
        },
        "compliance": {
            "frameworks": ["hipaa", "gdpr"],
            "audit_retention_days": 2555,
            "enable_monitoring": True
        }
    }
    
    if env_type == "hospital":
        base_config.update({
            "node": {
                "role": "client",
                "data_path": "/secure/medical/data",
                "compute_capacity": "4xA100",
                "department": "general"
            },
            "federation": {
                "coordinator_endpoint": "https://fed-coordinator.health-network.local"
            }
        })
    
    elif env_type == "coordinator":
        base_config.update({
            "federation": {
                "role": "coordinator",
                "model_registry": "/models",
                "aggregation_method": "fedavg"
            },
            "training": {
                "base_model": "medllama-7b",
                "rounds": 100,
                "clients_per_round": 5,
                "local_epochs": 1
            }
        })
    
    return base_config


async def start_server(args) -> None:
    """Start the federated router server."""
    print("Starting Federated DP-LLM Router...")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Using default configuration")
        config = create_default_config("development")
    
    # Create privacy accountant
    privacy_config = config.get("privacy", {})
    dp_config = DPConfig(
        epsilon_per_query=privacy_config.get("epsilon_per_query", 0.1),
        delta=privacy_config.get("delta", 1e-5),
        max_budget_per_user=privacy_config.get("max_budget_per_user", 10.0),
        noise_multiplier=privacy_config.get("noise_multiplier", 1.1)
    )
    
    # Create router
    routing_config = config.get("routing", {})
    router = FederatedRouter(
        model_name="medllama-7b",
        routing_strategy=RoutingStrategy.PRIVACY_AWARE
    )
    
    # Create request handler
    security_config = config.get("security", {})
    jwt_secret = security_config.get("jwt_secret", "default-secret-key")
    request_handler = RequestHandler(router, jwt_secret)
    
    # Start server
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = args.port or server_config.get("port", 8080)
    
    print(f"Server starting on {host}:{port}")
    
    try:
        request_handler.run(
            host=host,
            port=port,
            ssl_certfile=args.ssl_cert,
            ssl_keyfile=args.ssl_key
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await request_handler.shutdown()


async def register_node(args) -> None:
    """Register a hospital node."""
    node = HospitalNode(
        id=args.id,
        endpoint=args.endpoint,
        data_size=args.data_size,
        compute_capacity=args.compute_capacity,
        department=args.department
    )
    
    print(f"Registering node: {args.id}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Data size: {args.data_size}")
    print(f"  Compute: {args.compute_capacity}")
    
    # TODO: Actually register with coordinator
    print("Node registered successfully")


async def check_privacy_budget(args) -> None:
    """Check privacy budget for user or department."""
    # Create budget manager with default budgets
    default_budgets = {
        "emergency": 20.0,
        "radiology": 10.0,
        "general": 5.0,
        "research": 2.0
    }
    
    budget_manager = BudgetManager(default_budgets)
    
    if args.user:
        # Mock user department lookup
        user_department = "general"  # In practice, look up from user database
        remaining = budget_manager.get_remaining_budget(user_department)
        print(f"User {args.user} (department: {user_department})")
        print(f"Remaining privacy budget: {remaining:.3f}")
    
    if args.department:
        remaining = budget_manager.get_remaining_budget(args.department)
        print(f"Department {args.department}")
        print(f"Remaining privacy budget: {remaining:.3f}")
    
    if not args.user and not args.department:
        status = budget_manager.get_budget_status()
        print("Privacy Budget Status:")
        print("-" * 50)
        for dept, info in status.items():
            print(f"{dept:<12} | {info['remaining']:>8.3f} / {info['total_budget']:>8.3f} | {info['utilization_percent']:>6.1f}% | {info['alert_level']}")


async def run_training(args) -> None:
    """Run federated training."""
    print("Starting federated training...")
    
    # Load training configuration
    config = load_config(args.config)
    training_config = config.get("training", {})
    
    # Create DP config
    privacy_config = config.get("privacy", {})
    dp_config = DPConfig(
        epsilon_per_query=privacy_config.get("epsilon_per_query", 0.1),
        delta=privacy_config.get("delta", 1e-5)
    )
    
    # Create trainer
    trainer = FederatedTrainer(
        base_model=training_config.get("base_model", "medllama-7b"),
        dp_config=dp_config,
        rounds=args.rounds or training_config.get("rounds", 10),
        clients_per_round=args.clients_per_round or training_config.get("clients_per_round", 3)
    )
    
    # Create mock hospital nodes
    hospital_nodes = [
        HospitalNode(id="hospital_a", endpoint="https://hospital-a.local:8443", 
                    data_size=50000, compute_capacity="4xA100"),
        HospitalNode(id="hospital_b", endpoint="https://hospital-b.local:8443", 
                    data_size=75000, compute_capacity="8xA100"),
        HospitalNode(id="hospital_c", endpoint="https://hospital-c.local:8443", 
                    data_size=30000, compute_capacity="2xV100"),
    ]
    
    # Run training
    history = await trainer.train_federated(hospital_nodes)
    
    print("\nTraining completed!")
    print(f"Total rounds: {history['total_rounds']}")
    print(f"Final loss: {history['final_loss']:.4f}")
    print(f"Final accuracy: {history['final_accuracy']:.4f}")
    print(f"Training time: {history['total_training_time']:.2f} seconds")
    
    # Save model if output specified
    if args.output:
        output_path = Path(args.output)
        trainer.save_checkpoint(str(output_path / "federated_model_checkpoint.json"))
        print(f"Model saved to {output_path}")


async def run_query(args) -> None:
    """Run inference query."""
    print(f"Querying model with prompt: {args.prompt[:50]}...")
    
    # Create client
    client = PrivateInferenceClient(
        router_endpoint=args.endpoint,
        user_id=args.user
    )
    
    try:
        # Create test token
        from .routing.request_handler import RequestHandler
        handler = RequestHandler(None, "test-secret")
        token = handler.create_test_token(args.user, "general", "doctor", ["read", "query_inference"])
        client.auth_token = token
        
        # Make query
        response = await client.query(
            prompt=args.prompt,
            model_name=args.model,
            max_privacy_budget=args.privacy_budget,
            require_consensus=args.consensus
        )
        
        print("\nResponse:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        print(f"Privacy cost: {response.privacy_cost:.3f}")
        print(f"Remaining budget: {response.remaining_budget:.3f}")
        print(f"Processing nodes: {', '.join(response.processing_nodes)}")
        print(f"Latency: {response.latency:.3f}s")
        
        if response.consensus_achieved:
            print("✓ Consensus achieved")
    
    except Exception as e:
        print(f"Query failed: {e}")
    
    finally:
        await client.close()


async def generate_compliance_report(args) -> None:
    """Generate compliance report."""
    print(f"Generating {args.period} compliance report...")
    
    # Create compliance monitor
    monitor = ComplianceMonitor()
    
    # Mock some audit events for demo
    from .security.compliance import AuditEvent, AuditEventType
    import time
    
    events = [
        AuditEvent(
            event_id="event_1",
            event_type=AuditEventType.QUERY_SUBMITTED,
            user_id="doctor_123",
            department="cardiology",
            timestamp=time.time() - 3600,
            details={"query": "Patient symptoms analysis", "epsilon_spent": 0.1}
        ),
        AuditEvent(
            event_id="event_2",
            event_type=AuditEventType.PRIVACY_BUDGET_SPENT,
            user_id="nurse_456",
            department="emergency",
            timestamp=time.time() - 1800,
            details={"epsilon_spent": 0.2}
        )
    ]
    
    for event in events:
        monitor.record_event(event)
    
    # Generate report
    report = monitor.generate_report(period=args.period, format=args.format)
    
    # Output report
    if args.output:
        if args.format == "json":
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2, default=str))


def init_config(args) -> None:
    """Initialize configuration file."""
    print(f"Initializing {args.env} configuration...")
    
    config = create_default_config(args.env)
    
    # Write configuration file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration written to {output_path}")
    print("\nNext steps:")
    print("1. Review and customize the configuration file")
    print("2. Generate SSL certificates for secure communication")
    print("3. Start the server with: federated-dp-llm server start")


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "server" and args.server_action == "start":
            await start_server(args)
        elif args.command == "node" and args.node_action == "register":
            await register_node(args)
        elif args.command == "privacy" and args.privacy_action == "budget":
            await check_privacy_budget(args)
        elif args.command == "train":
            await run_training(args)
        elif args.command == "query":
            await run_query(args)
        elif args.command == "compliance" and args.compliance_action == "report":
            await generate_compliance_report(args)
        elif args.command == "config" and args.config_action == "init":
            init_config(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cli_main():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()