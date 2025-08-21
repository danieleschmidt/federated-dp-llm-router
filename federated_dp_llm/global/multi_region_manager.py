"""
Multi-Region Manager for Global Federated DP-LLM Router
Provides comprehensive multi-region support with intelligent routing and failover.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime, timezone


class RegionStatus(Enum):
    """Region operational status"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class DataResidency(Enum):
    """Data residency requirements"""
    STRICT = "strict"           # Data must remain in region
    FLEXIBLE = "flexible"       # Data can cross regions with controls
    UNRESTRICTED = "unrestricted"  # No restrictions


@dataclass
class RegionConfig:
    """Configuration for a specific region"""
    region_id: str
    region_name: str
    country_code: str
    data_center_locations: List[str]
    compliance_requirements: List[str]
    data_residency: DataResidency
    privacy_regulations: List[str]
    latency_threshold_ms: int = 100
    capacity_limit: int = 1000
    timezone: str = "UTC"
    languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class RegionMetrics:
    """Real-time metrics for a region"""
    region_id: str
    status: RegionStatus
    current_load: int = 0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    availability: float = 100.0
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capacity_utilization: float = 0.0


@dataclass
class GlobalRequest:
    """Request with global routing requirements"""
    request_id: str
    user_location: Dict[str, Any]
    data_classification: str
    compliance_requirements: List[str]
    preferred_regions: List[str] = field(default_factory=list)
    exclude_regions: List[str] = field(default_factory=list)
    max_latency_ms: int = 200
    data_residency_requirement: DataResidency = DataResidency.FLEXIBLE


class MultiRegionManager:
    """
    Advanced multi-region manager with intelligent routing,
    compliance enforcement, and automatic failover.
    """
    
    def __init__(self):
        self.regions: Dict[str, RegionConfig] = {}
        self.region_metrics: Dict[str, RegionMetrics] = {}
        self.routing_policies: Dict[str, Any] = {}
        self.compliance_matrix: Dict[str, List[str]] = {}
        
        # Initialize default regions
        self._initialize_default_regions()
        self._initialize_compliance_matrix()
    
    def _initialize_default_regions(self):
        """Initialize default global regions"""
        default_regions = [
            RegionConfig(
                region_id="us-east-1",
                region_name="US East (Virginia)",
                country_code="US",
                data_center_locations=["Virginia", "North Carolina"],
                compliance_requirements=["HIPAA", "SOC2", "CCPA"],
                data_residency=DataResidency.FLEXIBLE,
                privacy_regulations=["CCPA", "HIPAA"],
                latency_threshold_ms=50,
                capacity_limit=2000,
                timezone="America/New_York",
                languages=["en", "es"]
            ),
            RegionConfig(
                region_id="eu-west-1",
                region_name="EU West (Ireland)",
                country_code="IE",
                data_center_locations=["Dublin"],
                compliance_requirements=["GDPR", "ISO27001"],
                data_residency=DataResidency.STRICT,
                privacy_regulations=["GDPR"],
                latency_threshold_ms=60,
                capacity_limit=1500,
                timezone="Europe/Dublin",
                languages=["en", "de", "fr", "es", "it"]
            ),
            RegionConfig(
                region_id="ap-southeast-1",
                region_name="Asia Pacific (Singapore)",
                country_code="SG",
                data_center_locations=["Singapore"],
                compliance_requirements=["PDPA", "ISO27001"],
                data_residency=DataResidency.FLEXIBLE,
                privacy_regulations=["PDPA"],
                latency_threshold_ms=80,
                capacity_limit=1200,
                timezone="Asia/Singapore",
                languages=["en", "zh", "ms", "th"]
            ),
            RegionConfig(
                region_id="ca-central-1",
                region_name="Canada Central (Toronto)",
                country_code="CA",
                data_center_locations=["Toronto"],
                compliance_requirements=["PIPEDA", "SOC2"],
                data_residency=DataResidency.STRICT,
                privacy_regulations=["PIPEDA"],
                latency_threshold_ms=70,
                capacity_limit=800,
                timezone="America/Toronto",
                languages=["en", "fr"]
            )
        ]
        
        for region in default_regions:
            self.regions[region.region_id] = region
            self.region_metrics[region.region_id] = RegionMetrics(
                region_id=region.region_id,
                status=RegionStatus.ACTIVE
            )
    
    def _initialize_compliance_matrix(self):
        """Initialize compliance requirements matrix"""
        self.compliance_matrix = {
            "healthcare": ["HIPAA", "GDPR", "PIPEDA"],
            "financial": ["SOX", "PCI-DSS", "GDPR"],
            "government": ["FedRAMP", "IL4", "GDPR"],
            "education": ["FERPA", "GDPR", "PIPEDA"],
            "general": ["SOC2", "ISO27001"]
        }
    
    async def route_request(self, request: GlobalRequest) -> Tuple[str, Dict[str, Any]]:
        """
        Route request to optimal region based on multiple factors
        """
        # 1. Filter regions by compliance requirements
        compliant_regions = self._filter_by_compliance(request)
        
        # 2. Apply data residency constraints
        residency_compliant = self._filter_by_data_residency(request, compliant_regions)
        
        # 3. Check region availability and capacity
        available_regions = self._filter_by_availability(residency_compliant)
        
        # 4. Calculate optimal region based on latency, load, and preferences
        optimal_region = self._select_optimal_region(request, available_regions)
        
        if not optimal_region:
            raise Exception("No suitable region available for request")
        
        routing_decision = {
            "selected_region": optimal_region,
            "routing_factors": {
                "compliance_filtered": len(compliant_regions),
                "residency_filtered": len(residency_compliant),
                "available_regions": len(available_regions),
                "selection_criteria": self._get_selection_criteria(request, optimal_region)
            },
            "fallback_regions": self._get_fallback_regions(request, available_regions, optimal_region)
        }
        
        return optimal_region, routing_decision
    
    def _filter_by_compliance(self, request: GlobalRequest) -> List[str]:
        """Filter regions by compliance requirements"""
        compliant_regions = []
        
        for region_id, region in self.regions.items():
            if self._meets_compliance_requirements(region, request.compliance_requirements):
                compliant_regions.append(region_id)
        
        return compliant_regions
    
    def _meets_compliance_requirements(self, region: RegionConfig, requirements: List[str]) -> bool:
        """Check if region meets compliance requirements"""
        if not requirements:
            return True
        
        region_compliance = set(region.compliance_requirements)
        required_compliance = set(requirements)
        
        return required_compliance.issubset(region_compliance)
    
    def _filter_by_data_residency(self, request: GlobalRequest, regions: List[str]) -> List[str]:
        """Filter regions by data residency requirements"""
        if request.data_residency_requirement == DataResidency.UNRESTRICTED:
            return regions
        
        filtered_regions = []
        user_country = request.user_location.get("country_code", "")
        
        for region_id in regions:
            region = self.regions[region_id]
            
            if request.data_residency_requirement == DataResidency.STRICT:
                # Data must stay in same country/region
                if region.country_code == user_country or region_id in request.preferred_regions:
                    filtered_regions.append(region_id)
            else:  # FLEXIBLE
                # Allow cross-border with appropriate controls
                if (region.country_code == user_country or 
                    region_id in request.preferred_regions or
                    region.data_residency != DataResidency.STRICT):
                    filtered_regions.append(region_id)
        
        return filtered_regions if filtered_regions else regions
    
    def _filter_by_availability(self, regions: List[str]) -> List[str]:
        """Filter regions by availability and capacity"""
        available_regions = []
        
        for region_id in regions:
            if region_id in self.region_metrics:
                metrics = self.region_metrics[region_id]
                region_config = self.regions[region_id]
                
                # Check status and capacity
                if (metrics.status in [RegionStatus.ACTIVE, RegionStatus.DEGRADED] and
                    metrics.current_load < region_config.capacity_limit * 0.9):
                    available_regions.append(region_id)
        
        return available_regions
    
    def _select_optimal_region(self, request: GlobalRequest, regions: List[str]) -> Optional[str]:
        """Select optimal region using multi-criteria scoring"""
        if not regions:
            return None
        
        if len(regions) == 1:
            return regions[0]
        
        region_scores = {}
        
        for region_id in regions:
            score = self._calculate_region_score(request, region_id)
            region_scores[region_id] = score
        
        # Return region with highest score
        return max(region_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_region_score(self, request: GlobalRequest, region_id: str) -> float:
        """Calculate comprehensive region score"""
        region = self.regions[region_id]
        metrics = self.region_metrics[region_id]
        
        score = 0.0
        
        # 1. Latency score (40% weight)
        estimated_latency = self._estimate_latency(request.user_location, region)
        if estimated_latency <= request.max_latency_ms:
            latency_score = max(0, 1 - (estimated_latency / request.max_latency_ms))
            score += latency_score * 0.4
        
        # 2. Capacity score (25% weight)
        capacity_score = 1 - (metrics.current_load / region.capacity_limit)
        score += capacity_score * 0.25
        
        # 3. Performance score (20% weight)
        performance_score = (metrics.availability / 100) * (1 - metrics.error_rate)
        score += performance_score * 0.2
        
        # 4. Preference score (15% weight)
        if region_id in request.preferred_regions:
            score += 0.15
        elif region_id in request.exclude_regions:
            score -= 0.15
        
        return max(0, min(1, score))
    
    def _estimate_latency(self, user_location: Dict[str, Any], region: RegionConfig) -> float:
        """Estimate latency based on geographical distance"""
        # Simplified latency estimation based on region
        base_latencies = {
            "us-east-1": {"US": 20, "CA": 40, "EU": 80, "AP": 150},
            "eu-west-1": {"EU": 25, "US": 80, "CA": 90, "AP": 120},
            "ap-southeast-1": {"AP": 30, "US": 150, "EU": 120, "CA": 160},
            "ca-central-1": {"CA": 15, "US": 35, "EU": 85, "AP": 155}
        }
        
        user_region = user_location.get("region", "US")
        return base_latencies.get(region.region_id, {}).get(user_region, 100)
    
    def _get_selection_criteria(self, request: GlobalRequest, region_id: str) -> Dict[str, Any]:
        """Get detailed selection criteria for chosen region"""
        region = self.regions[region_id]
        metrics = self.region_metrics[region_id]
        
        return {
            "latency_ms": self._estimate_latency(request.user_location, region),
            "capacity_utilization": metrics.current_load / region.capacity_limit,
            "availability": metrics.availability,
            "compliance_match": len(set(region.compliance_requirements) & set(request.compliance_requirements)),
            "preferred_region": region_id in request.preferred_regions
        }
    
    def _get_fallback_regions(self, request: GlobalRequest, available_regions: List[str], 
                            primary_region: str) -> List[str]:
        """Get ordered list of fallback regions"""
        fallback_regions = [r for r in available_regions if r != primary_region]
        
        # Sort by score
        fallback_scores = [(r, self._calculate_region_score(request, r)) for r in fallback_regions]
        fallback_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in fallback_scores[:3]]  # Top 3 fallbacks
    
    async def update_region_metrics(self, region_id: str, metrics_update: Dict[str, Any]):
        """Update real-time metrics for a region"""
        if region_id in self.region_metrics:
            metrics = self.region_metrics[region_id]
            
            for key, value in metrics_update.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            metrics.last_health_check = datetime.now(timezone.utc)
    
    async def handle_region_failover(self, failed_region: str, active_requests: List[GlobalRequest]):
        """Handle failover when a region becomes unavailable"""
        # Mark region as offline
        if failed_region in self.region_metrics:
            self.region_metrics[failed_region].status = RegionStatus.OFFLINE
        
        # Redistribute active requests
        redistribution_results = []
        
        for request in active_requests:
            try:
                new_region, routing_decision = await self.route_request(request)
                redistribution_results.append({
                    "request_id": request.request_id,
                    "original_region": failed_region,
                    "new_region": new_region,
                    "success": True
                })
            except Exception as e:
                redistribution_results.append({
                    "request_id": request.request_id,
                    "original_region": failed_region,
                    "error": str(e),
                    "success": False
                })
        
        return redistribution_results
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global system status"""
        total_capacity = sum(r.capacity_limit for r in self.regions.values())
        total_load = sum(m.current_load for m in self.region_metrics.values())
        
        active_regions = sum(1 for m in self.region_metrics.values() 
                           if m.status == RegionStatus.ACTIVE)
        
        avg_availability = sum(m.availability for m in self.region_metrics.values()) / len(self.region_metrics)
        
        return {
            "total_regions": len(self.regions),
            "active_regions": active_regions,
            "global_capacity_utilization": total_load / total_capacity if total_capacity > 0 else 0,
            "average_availability": avg_availability,
            "supported_countries": list(set(r.country_code for r in self.regions.values())),
            "supported_languages": list(set().union(*(r.languages for r in self.regions.values()))),
            "compliance_frameworks": list(set().union(*(r.compliance_requirements for r in self.regions.values()))),
            "region_status": {
                region_id: {
                    "status": metrics.status.value,
                    "load": metrics.current_load,
                    "availability": metrics.availability
                }
                for region_id, metrics in self.region_metrics.items()
            }
        }


# Global instance
global_region_manager = MultiRegionManager()