"""
Synthetic API LLM Client Module
Handles communication with the Synthetic API for LLM-powered strategy.
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Parsed LLM response."""
    action: str
    reasoning: str
    priority: int
    target: Dict[str, Any]
    raw_content: str
    model: str
    tokens_used: Dict[str, int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_content: str, model: str) -> "LLMResponse":
        """Create from dictionary with defaults."""
        return cls(
            action=data.get("action", "PATROL"),
            reasoning=data.get("reasoning", "No reasoning provided"),
            priority=data.get("priority", 5),
            target=data.get("target", {}),
            raw_content=raw_content,
            model=model,
            tokens_used=data.get("tokens_used", {})
        )


class LLMClient:
    """
    Client for Synthetic API (OpenAI-compatible).
    Supports async calls with retry logic.
    """

    DEFAULT_API_URL = "https://api.synthetic.new/openai/v1"
    DEFAULT_MODEL = "hf:deepseek-ai/DeepSeek-V3"
    DEFAULT_TIMEOUT = 5.0  # seconds
    MAX_RETRIES = 2
    RETRY_DELAY = 1.0  # seconds

    SYSTEM_PROMPT = """You are a strategic AI brain for a game automation system.
Analyze the game state and determine the best high-level action.

Available Actions:
- HUNT: Hunt for enemies/targets
- GATHER: Collect resources
- GO_TO_MARKET: Travel to market/vendor
- RETREAT: Fall back to safe location
- PATROL: Patrol known area
- EXPLORE: Explore new areas

Priority Guidelines (1=highest urgency, 10=lowest):
- Priority 1-2: Critical situations (low health, immediate threats)
- Priority 3-4: High value targets or opportunities
- Priority 5-6: Routine exploration/patrolling
- Priority 7-10: Passive monitoring

Target Schema (if applicable):
- type: "enemy", "resource", "market", "location"
- position: {"x": float, "y": float, "z": float}
- label: string (name of target)

Respond ONLY with valid JSON in this format:
{
  "action": "ACTION_NAME",
  "reasoning": "explanation of choice",
  "priority": 1-10,
  "target": {
    "type": "enemy|resource|market|location",
    "position": {"x": 0, "y": 0, "z": 0}

  }
}

Do NOT include any text outside the JSON object."""

    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Synthetic API key.
            api_url: API endpoint URL.
            model: Model identifier to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Rate limiting tracking
        self._rate_limit_backoff_until: float = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self):
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._headers
            )
            logger.info(f"[LLM] Connected to {self.api_url}")

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("[LLM] Client closed")

    def _is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return asyncio.get_event_loop().time() < self._rate_limit_backoff_until

    def _set_rate_limit_backoff(self, seconds: int = 60):
        """Set rate limit backoff."""
        self._rate_limit_backoff_until = asyncio.get_event_loop().time() + seconds
        logger.warning(f"[LLM] Rate limited, backing off for {seconds}s")

    async def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Send chat completion request to LLM.

        Args:
            prompt: User prompt to send.
            system_prompt: System prompt override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            LLMResponse with parsed action and metadata.
        """
        # Check rate limit
        if self._is_rate_limited():
            raise Exception("LLM rate limited, using fallback")

        # Ensure client is connected
        if self._client is None:
            await self.connect()

        system = system_prompt or self.SYSTEM_PROMPT

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                return self._parse_response(data, prompt)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit error
                    logger.warning("[LLM] Rate limit hit")
                    self._set_rate_limit_backoff()
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise Exception("LLM rate limit exceeded")

                logger.error(f"[LLM] HTTP error {e.response.status_code}: {e}")
                last_error = e

            except httpx.TimeoutException as e:
                logger.warning(f"[LLM] Request timeout: {e}")
                last_error = e
                if attempt < self.max_retries:
                    continue

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"[LLM] Parse error: {e}")
                last_error = e
                break  # No retry for parse errors

            except Exception as e:
                logger.error(f"[LLM] Unexpected error: {e}")
                last_error = e
                break

            # Wait before retry
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay)

        # All retries failed
        raise Exception(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")

    def _parse_response(self, data: Dict[str, Any], original_prompt: str) -> LLMResponse:
        """
        Parse LLM API response.

        Args:
            data: Raw response from API.
            original_prompt: Original prompt for reference.

        Returns:
            Parsed LLMResponse.
        """
        try:
            # Extract content
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            content = choices[0].get("message", {}).get("content", "")

            # Extract token usage
            usage = data.get("usage", {})
            tokens_used = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

            # Parse JSON from content
            # Handle potential markdown code blocks
            if content.strip().startswith("```json"):
                content = content.strip()[7:]  # Remove ```json
                if content.strip().endswith("```"):
                    content = content.strip()[:-3]  # Remove ```
            elif content.strip().startswith("```"):
                content = content.strip()[3:]  # Remove ```
                if content.strip().endswith("```"):
                    content = content.strip()[:-3]  # Remove ```

            parsed = json.loads(content.strip())

            # Validate required fields
            if "action" not in parsed:
                parsed["action"] = "PATROL"
            if "reasoning" not in parsed:
                parsed["reasoning"] = "No reasoning provided"
            if "priority" not in parsed:
                parsed["priority"] = 5
            if "target" not in parsed:
                parsed["target"] = {}

            return LLMResponse.from_dict(
                parsed,
                content,
                data.get("model", self.model)
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[LLM] Parse error, using fallback: {e}")
            # Return safe default
            return LLMResponse(
                action="PATROL",
                reasoning=f"Parse error, patrolling: {str(e)}",
                priority=6,
                target={},
                raw_content=content if 'content' in locals() else "",
                model=data.get("model", self.model),
                tokens_used={}
            )

    async def get_available_models(self) -> List[str]:
        """
        Get list of available models from the API.

        Returns:
            List of model IDs.
        """
        if self._client is None:
            await self.connect()

        try:
            response = await self._client.get(f"{self.api_url}/models")
            response.raise_for_status()
            data = response.json()

            models = [m.get("id", "") for m in data.get("data", [])]
            logger.info(f"[LLM] Available models: {models}")
            return models

        except Exception as e:
            logger.error(f"[LLM] Failed to fetch models: {e}")
            return [self.model]

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the LLM API.

        Returns:
            Connection test result.
        """
        try:
            models = await self.get_available_models()
            if self.model not in models:
                logger.warning(f"[LLM] Model {self.model} not in available list")

            # Test simple chat
            start_time = asyncio.get_event_loop().time()
            response = await self.chat_completion(
                "Test connection. Respond with JSON: {\"action\": \"TEST\", \"reasoning\": \"Connection successful\", \"priority\": 1}"
            )
            latency = (asyncio.get_event_loop().time() - start_time) * 1000

            return {
                "success": True,
                "model": self.model,
                "latency_ms": round(latency, 2),
                "tokens_used": response.tokens_used,
                "available_models": len(models)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_strategy_prompt(self, state: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Build strategy prompt from game state.

        Args:
            state: Current game state.
            context: Strategy context (spatial memory, etc.).

        Returns:
            Formatted prompt string.
        """
        # Describe current situation
        situation = f"""Current Game State:
- Security Level: {state.get('security_tier', 'UNKNOWN')}
- Player Health: {state.get('player', {}).get('health_percent', 0)}%
- In Combat: {state.get('player', {}).get('is_in_combat', False)}
- Inventory Full: {context.get('inventory_state', {}).get('full_percent', 0)}%
- Active Targets: {len(state.get('active_targets', []))}

Detections:"""

        for detection in state.get("detections", [])[:5]:  # Top 5 detections
            situation += f"\n- {detection.get('label', 'Unknown')} ({detection.get('detection_type', 'other')}) - Confidence: {detection.get('confidence', 0):.2f}"
            if detection.get('distance'):
                situation += f" - Distance: {detection['distance']:.1f}m"
            if detection.get('health_percent'):
                situation += f" - Health: {detection['health_percent']}%"

        # Describe spatial memory
        memory_info = "\n\nKnown Locations:"
        for entity in context.get("spatial_memory", [])[:10]:  # Top 10 known entities
            memory_info += f"\n- {entity.get('type', 'unknown')}: {entity.get('label', 'Unknown')} at {entity.get('distance', 0):.1f}m"

        # Current strategy
        current_strategy = context.get("current_strategy")
        strategy_info = f"\n\nCurrent Strategy: {current_strategy.get('action', 'None')}" if current_strategy else "\n\nCurrent Strategy: None"

        prompt = f"""{situation}{memory_info}{strategy_info}

Determine the best action given the current situation.
"""
        return prompt


def create_llm_client(
    api_key: str,
    api_url: str = LLMClient.DEFAULT_API_URL,
    model: str = LLMClient.DEFAULT_MODEL
) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        api_key: Synthetic API key.
        api_url: API endpoint URL.
        model: Model identifier.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(
        api_key=api_key,
        api_url=api_url,
        model=model
    )
