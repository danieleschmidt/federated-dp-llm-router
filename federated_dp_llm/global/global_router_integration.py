"""
Global Router Integration for Federated DP-LLM Router
Integrates multi-region management and i18n capabilities with the core routing system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .multi_region_manager import global_region_manager, GlobalRequest, DataResidency
from .i18n_manager import global_i18n_manager, TranslationRequest, Language


@dataclass
class GlobalRoutingRequest:
    """Enhanced routing request with global capabilities"""
    request_id: str
    user_id: str
    user_location: Dict[str, Any]
    preferred_language: str = "en"
    content: str = ""
    
    # Privacy and compliance
    data_classification: str = "general"
    compliance_requirements: List[str] = field(default_factory=list)
    data_residency_requirement: DataResidency = DataResidency.FLEXIBLE
    
    # Performance requirements
    max_latency_ms: int = 200
    priority: str = "normal"  # low, normal, high, critical
    
    # Regional preferences
    preferred_regions: List[str] = field(default_factory=list)
    exclude_regions: List[str] = field(default_factory=list)
    
    # I18n requirements
    require_translation: bool = False
    target_languages: List[str] = field(default_factory=list)
    cultural_adaptation: bool = True
    formality_level: str = "neutral"


@dataclass
class GlobalRoutingResponse:
    """Enhanced routing response with global metadata"""
    request_id: str
    selected_region: str
    processing_time_ms: float
    
    # Routing details
    routing_decision: Dict[str, Any]
    fallback_regions: List[str]
    
    # I18n details
    detected_language: str
    language_confidence: float
    translation_applied: bool = False
    cultural_adaptations: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_latency_ms: float = 0.0
    region_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    
    # Compliance status
    compliance_status: str = "compliant"
    privacy_level: str = "standard"
    data_residency_status: str = "satisfied"


class GlobalRouterIntegration:
    """
    Comprehensive global router integration that combines
    multi-region routing with internationalization capabilities.
    """
    
    def __init__(self):
        self.region_manager = global_region_manager
        self.i18n_manager = global_i18n_manager
        self.routing_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        
    async def route_global_request(self, request: GlobalRoutingRequest) -> GlobalRoutingResponse:
        """
        Route request with full global capabilities including
        region selection, language detection, and cultural adaptation.
        """
        start_time = time.time()
        
        # 1. Language Detection and Analysis
        detected_language, confidence = await self._detect_and_analyze_language(request)
        
        # 2. Prepare region routing request
        region_request = self._prepare_region_request(request)
        
        # 3. Route to optimal region
        selected_region, routing_decision = await self.region_manager.route_request(region_request)
        
        # 4. Apply internationalization if needed
        translation_applied, cultural_adaptations, translation_time = await self._apply_i18n(request, detected_language)
        
        # 5. Calculate performance metrics
        total_time = (time.time() - start_time) * 1000
        region_latency = routing_decision.get("routing_factors", {}).get("latency_ms", 0)
        
        # 6. Determine compliance and privacy status
        compliance_status, privacy_level, residency_status = self._assess_compliance(request, selected_region)
        
        # 7. Build comprehensive response
        response = GlobalRoutingResponse(
            request_id=request.request_id,
            selected_region=selected_region,
            processing_time_ms=total_time,
            routing_decision=routing_decision,
            fallback_regions=routing_decision.get("fallback_regions", []),
            detected_language=detected_language,
            language_confidence=confidence,
            translation_applied=translation_applied,
            cultural_adaptations=cultural_adaptations,
            total_latency_ms=total_time,
            region_latency_ms=region_latency,
            translation_latency_ms=translation_time,
            compliance_status=compliance_status,
            privacy_level=privacy_level,
            data_residency_status=residency_status
        )
        
        # 8. Update performance metrics
        await self._update_performance_metrics(selected_region, response)
        
        return response
    
    async def _detect_and_analyze_language(self, request: GlobalRoutingRequest) -> Tuple[str, float]:
        """Detect and analyze language requirements"""
        if request.content:
            detected_language, confidence = self.i18n_manager.detect_language(request.content)
            
            # Override with user preference if specified and confidence is low
            if confidence < 0.8 and request.preferred_language:
                detected_language = request.preferred_language
                confidence = 0.9  # High confidence in user preference
        else:
            detected_language = request.preferred_language or "en"
            confidence = 0.9
        
        return detected_language, confidence
    
    def _prepare_region_request(self, request: GlobalRoutingRequest) -> GlobalRequest:
        """Prepare region routing request from global request"""
        return GlobalRequest(
            request_id=request.request_id,
            user_location=request.user_location,
            data_classification=request.data_classification,
            compliance_requirements=request.compliance_requirements,
            preferred_regions=request.preferred_regions,
            exclude_regions=request.exclude_regions,
            max_latency_ms=request.max_latency_ms,
            data_residency_requirement=request.data_residency_requirement
        )
    
    async def _apply_i18n(self, request: GlobalRoutingRequest, 
                         detected_language: str) -> Tuple[bool, List[str], float]:
        """Apply internationalization processing"""
        start_time = time.time()
        translation_applied = False
        cultural_adaptations = []
        
        if request.require_translation and request.target_languages:
            # Apply translations for each target language
            for target_lang in request.target_languages:
                if target_lang != detected_language:
                    translation_request = TranslationRequest(
                        text=request.content,
                        source_language=detected_language,
                        target_language=target_lang,
                        context="healthcare",
                        formality_level=request.formality_level,
                        domain="healthcare"
                    )
                    
                    await self.i18n_manager.translate_text(translation_request)
                    translation_applied = True
        
        if request.cultural_adaptation:
            # Apply cultural adaptations
            if detected_language in self.i18n_manager.cultural_contexts:
                context = self.i18n_manager.cultural_contexts[detected_language]
                cultural_adaptations = context.cultural_considerations.copy()
        
        translation_time = (time.time() - start_time) * 1000
        return translation_applied, cultural_adaptations, translation_time
    
    def _assess_compliance(self, request: GlobalRoutingRequest, 
                          selected_region: str) -> Tuple[str, str, str]:
        """Assess compliance, privacy, and data residency status"""
        # Get region configuration
        region_config = self.region_manager.regions.get(selected_region)
        if not region_config:
            return "unknown", "unknown", "unknown"
        
        # Check compliance status
        compliance_status = "compliant"
        if request.compliance_requirements:
            region_compliance = set(region_config.compliance_requirements)
            required_compliance = set(request.compliance_requirements)
            if not required_compliance.issubset(region_compliance):
                compliance_status = "partial"
        
        # Determine privacy level
        privacy_level = "standard"
        if "GDPR" in region_config.privacy_regulations:
            privacy_level = "enhanced"
        elif "HIPAA" in region_config.privacy_regulations:
            privacy_level = "healthcare"
        
        # Check data residency status
        residency_status = "satisfied"
        user_country = request.user_location.get("country_code", "")
        
        if request.data_residency_requirement == DataResidency.STRICT:
            if region_config.country_code != user_country:
                residency_status = "violation"
        elif request.data_residency_requirement == DataResidency.FLEXIBLE:
            if region_config.data_residency == DataResidency.STRICT and region_config.country_code != user_country:
                residency_status = "flexible_allowed"
        
        return compliance_status, privacy_level, residency_status
    
    async def _update_performance_metrics(self, region: str, response: GlobalRoutingResponse):
        """Update performance metrics for monitoring"""
        if region not in self.performance_metrics:
            self.performance_metrics[region] = []
        
        self.performance_metrics[region].append(response.total_latency_ms)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics[region]) > 100:
            self.performance_metrics[region] = self.performance_metrics[region][-100:]
    
    async def get_global_system_status(self) -> Dict[str, Any]:
        """Get comprehensive global system status"""
        # Get regional status
        regional_status = self.region_manager.get_global_status()
        
        # Get i18n status
        i18n_stats = self.i18n_manager.get_language_statistics()
        
        # Calculate performance metrics
        performance_summary = {}
        for region, latencies in self.performance_metrics.items():
            if latencies:
                performance_summary[region] = {
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "max_latency_ms": max(latencies),
                    "min_latency_ms": min(latencies),
                    "sample_count": len(latencies)
                }
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regional_status": regional_status,
            "i18n_capabilities": i18n_stats,
            "performance_metrics": performance_summary,
            "global_features": {
                "multi_region_routing": True,
                "intelligent_failover": True,
                "cultural_adaptation": True,
                "real_time_translation": True,
                "compliance_enforcement": True,
                "data_residency_control": True
            },
            "supported_regions": list(self.region_manager.regions.keys()),
            "supported_languages": [lang["code"] for lang in self.i18n_manager.get_supported_languages()],
            "cache_status": {
                "routing_cache_size": len(self.routing_cache),
                "performance_metrics_regions": len(self.performance_metrics)
            }
        }
    
    async def handle_global_failover(self, failed_region: str, 
                                   active_requests: List[GlobalRoutingRequest]) -> Dict[str, Any]:
        """Handle global failover with language and cultural considerations"""
        # Convert global requests to region requests for failover
        region_requests = [self._prepare_region_request(req) for req in active_requests]
        
        # Execute regional failover
        failover_results = await self.region_manager.handle_region_failover(failed_region, region_requests)
        
        # Apply language-specific notifications
        notifications = {}
        for request in active_requests:
            lang = request.preferred_language or "en"
            message = await self.i18n_manager.get_localized_message(
                "error_message", 
                lang,
                {"region": failed_region}
            )
            notifications[request.request_id] = {
                "language": lang,
                "message": message,
                "requires_translation": request.require_translation
            }
        
        return {
            "failover_results": failover_results,
            "localized_notifications": notifications,
            "failed_region": failed_region,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def optimize_global_routing(self) -> Dict[str, Any]:
        """Optimize global routing based on performance metrics and language patterns"""
        optimization_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimizations_applied": [],
            "performance_improvements": {}
        }
        
        # 1. Analyze regional performance patterns
        region_performance = {}
        for region, latencies in self.performance_metrics.items():
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                region_performance[region] = avg_latency
        
        # 2. Identify underperforming regions
        if region_performance:
            avg_global_latency = sum(region_performance.values()) / len(region_performance)
            slow_regions = [r for r, lat in region_performance.items() if lat > avg_global_latency * 1.5]
            
            if slow_regions:
                optimization_results["optimizations_applied"].append({
                    "type": "regional_performance_adjustment",
                    "affected_regions": slow_regions,
                    "action": "increased_health_check_frequency"
                })
        
        # 3. Optimize language routing patterns
        supported_languages = self.i18n_manager.get_supported_languages()
        regional_languages = {}
        
        for region_id in self.region_manager.regions:
            regional_languages[region_id] = self.i18n_manager.get_regional_languages(
                self.region_manager.regions[region_id].country_code
            )
        
        optimization_results["language_optimization"] = {
            "regional_language_mapping": regional_languages,
            "total_language_coverage": len(supported_languages)
        }
        
        return optimization_results


# Global instance
global_router_integration = GlobalRouterIntegration()