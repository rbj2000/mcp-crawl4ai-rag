"""HuggingFace AI provider implementation with reranking support"""

import asyncio
import logging
import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import numpy as np

from ..base import (
    RerankingProvider,
    RerankingResult,
    HealthStatus,
    AIProvider
)

logger = logging.getLogger(__name__)


class HuggingFaceRerankingProvider(RerankingProvider):
    """HuggingFace reranking provider using CrossEncoder models"""
    
    # Common HuggingFace CrossEncoder models for reranking
    MODEL_CONFIGS = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": {
            "size": "22.7M",
            "description": "Fast and efficient cross-encoder for MS MARCO dataset",
            "max_length": 512
        },
        "cross-encoder/ms-marco-MiniLM-L-12-v2": {
            "size": "33.4M", 
            "description": "Larger cross-encoder with better performance",
            "max_length": 512
        },
        "cross-encoder/ms-marco-TinyBERT-L-2-v2": {
            "size": "17.7M",
            "description": "Smallest and fastest cross-encoder model",
            "max_length": 512
        },
        "cross-encoder/stsb-roberta-large": {
            "size": "355M",
            "description": "RoBERTa-large based cross-encoder for semantic similarity",
            "max_length": 512
        },
        "cross-encoder/nli-deberta-v3-large": {
            "size": "434M",
            "description": "DeBERTa-large based cross-encoder for natural language inference",
            "max_length": 512
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace reranking provider
        
        Args:
            config: Configuration dictionary with HuggingFace settings
        """
        super().__init__(config)
        self.model = None
        self.model_name = config.get("default_reranking_model", self.default_reranking_model)
        
        # Authentication settings
        self.api_token = config.get("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN")
        
        # Performance settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 1.0)
        self.max_retry_delay = config.get("max_retry_delay", 30.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)
        
        # Model settings
        self._max_results_count = config.get("reranking_max_results", 100)
        self.device = config.get("device", "cpu")  # cpu or cuda
        self.batch_size = config.get("batch_size", 32)
        
        # Performance monitoring
        self._request_count = 0
        self._total_results_processed = 0
        self._error_count = 0
        self._model_load_time = 0.0
        
    def _validate_config(self) -> None:
        """Validate HuggingFace-specific configuration"""
        model_name = self.config.get("default_reranking_model", "")
        if model_name and not model_name.startswith("cross-encoder/"):
            logger.warning(
                f"HuggingFace reranking model '{model_name}' doesn't follow "
                "expected naming pattern. Consider using a cross-encoder model."
            )
        
        # Validate device setting
        device = self.config.get("device", "cpu")
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        
        # Validate batch size
        batch_size = self.config.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        
        if batch_size > 128:
            logger.warning(f"Large batch size {batch_size} may cause memory issues")
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace CrossEncoder model"""
        try:
            start_time = time.time()
            
            # Import sentence-transformers here to avoid import errors if not installed
            try:
                from sentence_transformers import CrossEncoder
                self._CrossEncoder = CrossEncoder
            except ImportError:
                raise ConnectionError(
                    "sentence-transformers package not installed. "
                    "Install with: pip install sentence-transformers"
                )
            
            # Set HuggingFace authentication if token provided
            if self.api_token:
                try:
                    from huggingface_hub import login
                    login(token=self.api_token)
                    logger.info("HuggingFace authentication successful")
                except ImportError:
                    logger.warning("huggingface_hub not installed, skipping authentication")
                except Exception as e:
                    logger.warning(f"HuggingFace authentication failed: {e}")
            
            # Load the model
            logger.info(f"Loading HuggingFace CrossEncoder model: {self.model_name}")
            self.model = self._CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self._get_model_max_length()
            )
            
            self._model_load_time = time.time() - start_time
            logger.info(
                f"HuggingFace reranking provider initialized successfully "
                f"with model '{self.model_name}' in {self._model_load_time:.2f}s"
            )
            
            # Perform health check
            health = await self.health_check()
            if not health.is_healthy:
                raise ConnectionError(f"Health check failed: {health.error_message}")
                
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace reranking provider: {e}")
            self.model = None
            raise ConnectionError(f"Failed to initialize HuggingFace CrossEncoder: {e}")
    
    async def close(self) -> None:
        """Close the HuggingFace model"""
        if self.model:
            # Clean up model resources
            try:
                if hasattr(self.model, 'model'):
                    del self.model.model
                del self.model
                self.model = None
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except ImportError:
                        pass
                        
                logger.info("HuggingFace reranking provider closed")
            except Exception as e:
                logger.warning(f"Error during HuggingFace provider cleanup: {e}")
    
    async def health_check(self) -> HealthStatus:
        """Check if HuggingFace model is healthy and responsive"""
        if not self.model:
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.HUGGINGFACE.value,
                model=self.model_name,
                response_time_ms=0.0,
                error_message="Model not initialized"
            )
        
        start_time = time.time()
        try:
            # Test with a simple reranking request
            test_query = "test query"
            test_results = [
                {"content": "test document 1"},
                {"content": "test document 2"}
            ]
            
            await self.rerank_results(test_query, test_results)
            
            response_time_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                is_healthy=True,
                provider=AIProvider.HUGGINGFACE.value,
                model=self.model_name,
                response_time_ms=response_time_ms,
                metadata={
                    "device": self.device,
                    "model_load_time": self._model_load_time,
                    "max_length": self._get_model_max_length()
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.HUGGINGFACE.value,
                model=self.model_name,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> RerankingResult:
        """Rerank search results using HuggingFace CrossEncoder
        
        Args:  
            query: Search query for relevance scoring
            results: List of search results to rerank
            model: Optional model override (not supported, will use configured model)
            
        Returns:
            RerankingResult with reranked results and scores
        """
        if not self.model:
            raise ConnectionError("Model not initialized")
        
        # Validate input
        self.validate_reranking_input(query, results)
        
        if model and model != self.model_name:
            logger.warning(
                f"Model override '{model}' not supported. "
                f"Using configured model '{self.model_name}'"
            )
        
        start_time = time.time()
        
        try:
            # Prepare query-document pairs for CrossEncoder
            query_doc_pairs = []
            for result in results:
                content = result.get("content", "")
                if isinstance(content, str) and content.strip():
                    query_doc_pairs.append([query, content])
                else:
                    # Handle empty content with zero score
                    query_doc_pairs.append([query, "[empty]"])
            
            # Get relevance scores from CrossEncoder
            scores = await self._predict_with_retry(query_doc_pairs)
            
            # Create reranked results with scores
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, scores)):
                enhanced_result = result.copy()
                enhanced_result["rerank_score"] = float(score)
                enhanced_result["original_rank"] = i
                reranked_results.append(enhanced_result)
            
            # Sort by rerank score in descending order
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._request_count += 1
            self._total_results_processed += len(results)
            
            logger.debug(
                f"Reranked {len(results)} results in {processing_time_ms:.1f}ms "
                f"using model '{self.model_name}'"
            )
            
            return RerankingResult(
                results=reranked_results,
                model=self.model_name,
                provider=AIProvider.HUGGINGFACE.value,
                rerank_scores=scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                processing_time_ms=processing_time_ms,
                metadata={
                    "device": self.device,
                    "num_pairs": len(query_doc_pairs),
                    "batch_size": self.batch_size,
                    "max_length": self._get_model_max_length()
                }
            )
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Reranking failed: {e}")
            raise
    
    async def _predict_with_retry(self, query_doc_pairs: List[List[str]]) -> np.ndarray:
        """Predict relevance scores with retry logic"""
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                # Run prediction in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                scores = await loop.run_in_executor(
                    None,
                    self._predict_batch,
                    query_doc_pairs
                )
                return scores
                
            except Exception as e:
                self._error_count += 1
                
                if attempt < self.max_retries:
                    if self._is_retryable_error(e):
                        logger.warning(
                            f"Reranking prediction failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {retry_delay:.1f}s"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(
                            retry_delay * self.retry_exponential_base,
                            self.max_retry_delay
                        )
                        continue
                    else:
                        logger.error(f"Non-retryable error in reranking prediction: {e}")
                        raise
                else:
                    logger.error(f"Reranking prediction failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    def _predict_batch(self, query_doc_pairs: List[List[str]]) -> np.ndarray:
        """Synchronous batch prediction method"""
        if not self.model:
            raise ConnectionError("Model not initialized")
        
        # Process in batches to manage memory
        all_scores = []
        
        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch_end = min(i + self.batch_size, len(query_doc_pairs))
            batch_pairs = query_doc_pairs[i:batch_end]
            
            try:
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores)
            except Exception as e:
                logger.error(f"Batch prediction failed for batch {i//self.batch_size + 1}: {e}")
                # Return zero scores for this batch as fallback
                all_scores.extend([0.0] * len(batch_pairs))
        
        return np.array(all_scores)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()
        
        # Memory errors (could be temporary)
        if any(term in error_str for term in ["memory", "cuda out of memory", "allocation"]):
            return True
        
        # Network/connection errors (for model downloads)
        if any(term in error_str for term in ["connection", "timeout", "network"]):
            return True
        
        # HuggingFace specific errors
        if any(term in error_str for term in ["503", "502", "500", "rate limit"]):
            return True
        
        return False
    
    def _get_model_max_length(self) -> int:
        """Get maximum sequence length for the model"""
        return self.MODEL_CONFIGS.get(self.model_name, {}).get("max_length", 512)
    
    def supports_reranking(self) -> bool:
        """Check if provider supports reranking functionality"""
        return self.model is not None
    
    def get_reranking_models(self) -> List[str]:
        """Get list of available reranking models"""
        return list(self.MODEL_CONFIGS.keys())
    
    @property
    def default_reranking_model(self) -> str:
        """Default reranking model for HuggingFace provider"""
        return "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    @property
    def max_results_count(self) -> int:
        """Maximum number of results that can be reranked at once"""
        return self._max_results_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics"""
        return {
            "request_count": self._request_count,
            "total_results_processed": self._total_results_processed,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "model": self.model_name,
            "device": self.device,
            "model_load_time": self._model_load_time,
            "average_results_per_request": (
                self._total_results_processed / max(self._request_count, 1)
            )
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model_config = self.MODEL_CONFIGS.get(self.model_name, {})
        return {
            "name": self.model_name,
            "provider": AIProvider.HUGGINGFACE.value,
            "size": model_config.get("size", "Unknown"),
            "description": model_config.get("description", ""),
            "max_length": model_config.get("max_length", 512),
            "device": self.device,
            "batch_size": self.batch_size,
            "is_loaded": self.model is not None
        }