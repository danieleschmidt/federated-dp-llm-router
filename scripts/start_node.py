#!/usr/bin/env python3
"""
Hospital Node Startup Script

Starts a simulated hospital node for federated learning demonstration.
"""

import os
import asyncio
import time
from federated_dp_llm.federation.client import HospitalNode, FederatedLearningClient
from federated_dp_llm.routing.request_handler import RequestHandler
from federated_dp_llm.routing.load_balancer import FederatedRouter
from federated_dp_llm.core.privacy_accountant import DPConfig
from federated_dp_llm.monitoring.logging_config import setup_logging, LogConfig


async def main():
    """Start hospital node."""
    
    # Setup logging
    log_config = LogConfig(
        level="INFO",
        format="structured",
        output="both",
        log_file=f"/app/logs/node_{os.getenv('NODE_ID', 'unknown')}.log",
        enable_audit_logging=True
    )
    setup_logging(log_config)
    
    # Get environment configuration
    node_id = os.getenv('NODE_ID', 'hospital_node')
    node_endpoint = os.getenv('NODE_ENDPOINT', 'https://localhost:8443')
    coordinator_endpoint = os.getenv('COORDINATOR_ENDPOINT', 'https://localhost:8080')
    department = os.getenv('DEPARTMENT', 'general')
    data_size = int(os.getenv('DATA_SIZE', '10000'))
    compute_capacity = os.getenv('COMPUTE_CAPACITY', '4xA100')
    
    print(f"Starting hospital node: {node_id}")
    print(f"Department: {department}")
    print(f"Data size: {data_size}")
    print(f"Compute capacity: {compute_capacity}")
    
    # Create hospital node
    hospital_node = HospitalNode(
        id=node_id,
        endpoint=node_endpoint,
        data_size=data_size,
        compute_capacity=compute_capacity,
        department=department
    )
    
    # Create local router for node
    router = FederatedRouter(
        model_name="medllama-7b",
        num_shards=1  # Single shard for node
    )
    
    # Register this node with itself (for health checks)
    router.register_nodes([hospital_node])
    
    # Create request handler
    request_handler = RequestHandler(router, "node-secret-key")
    
    # Create federated learning client
    dp_config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
    fl_client = FederatedLearningClient(
        node=hospital_node,
        privacy_config=dp_config,
        coordinator_endpoint=coordinator_endpoint
    )
    
    # Register with coordinator
    try:
        success = await fl_client.register_with_coordinator()
        if success:
            print(f"Successfully registered with coordinator: {coordinator_endpoint}")
        else:
            print(f"Failed to register with coordinator: {coordinator_endpoint}")
    except Exception as e:
        print(f"Error registering with coordinator: {e}")
    
    # Start node server
    print(f"Starting node server on port 8443...")
    try:
        # Run server in background task
        server_task = asyncio.create_task(
            run_server(request_handler)
        )
        
        # Simulate federated learning participation
        training_task = asyncio.create_task(
            simulate_federated_training(fl_client)
        )
        
        # Wait for tasks
        await asyncio.gather(server_task, training_task)
        
    except KeyboardInterrupt:
        print("Shutting down node...")
        await request_handler.shutdown()
        await fl_client.close()


async def run_server(request_handler):
    """Run the node server."""
    import uvicorn
    
    config = uvicorn.Config(
        request_handler.app,
        host="0.0.0.0",
        port=8443,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


async def simulate_federated_training(fl_client):
    """Simulate participation in federated training rounds."""
    
    round_counter = 0
    
    while True:
        try:
            await asyncio.sleep(60)  # Wait 1 minute between rounds
            
            round_counter += 1
            round_id = f"round_{round_counter}"
            
            print(f"Simulating participation in {round_id}")
            
            # Create mock global model
            global_model = {
                "layer1": [[0.1, 0.2], [0.3, 0.4]],
                "layer2": [[0.5, 0.6]]
            }
            
            # Participate in training round
            model_update = await fl_client.participate_in_round(round_id, global_model)
            
            # Submit update (would normally send to coordinator)
            print(f"Generated model update for {round_id}")
            print(f"Privacy spent: {model_update.privacy_spent}")
            print(f"Training samples: {model_update.training_samples}")
            
        except Exception as e:
            print(f"Error in federated training simulation: {e}")
            await asyncio.sleep(30)  # Wait before retrying


if __name__ == "__main__":
    asyncio.run(main())