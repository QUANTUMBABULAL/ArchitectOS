"""
Worker framework and site-specific workers.

Workers own website-specific automation logic and communicate with online
AI systems through the generic browser operating system. The framework
contract (BaseWorker), the generic chat driver (WebChatWorker), the
provider registry, and the registry/dispatcher (WorkerManager) keep the
orchestration layer independent of any concrete website.
"""

from .auth import (
    AuthState,
    AuthStatus,
    challenge_prompt,
    expiry_notice,
    login_prompt,
    render_dashboard,
    state_glyph,
)
from .base_worker import (
    BaseWorker,
    WorkerConfig,
    WorkerHealth,
    WorkerQuery,
    WorkerResponse,
    WorkerState,
)
from .chat_site import ChatSiteConfig
from .chatgpt_worker import ChatGPTWorker, ChatGPTWorkerConfig
from .conversation import ConversationState
from .provider_workers import (
    ClaudeWorker,
    DeepSeekWorker,
    GeminiWorker,
    GrokWorker,
    MistralWorker,
    PerplexityWorker,
    QwenWorker,
)
from .providers import (
    CHATGPT_SITE,
    CLAUDE_SITE,
    DEEPSEEK_SITE,
    DEFAULT_PROVIDER,
    GEMINI_SITE,
    GROK_SITE,
    MISTRAL_SITE,
    PERPLEXITY_SITE,
    PROVIDER_SITES,
    QWEN_SITE,
    available_providers,
    build_registry,
    build_worker,
    default_disabled_providers,
    default_enabled_providers,
    get_site,
    parse_provider_list,
    verified_providers,
)
from .registry import DisableReason, ProviderRegistration, ProviderRegistry
from .web_chat_worker import WebChatWorker
from .worker_manager import WorkerManager

__all__ = [
    "AuthState",
    "AuthStatus",
    "BaseWorker",
    "CHATGPT_SITE",
    "CLAUDE_SITE",
    "ChatGPTWorker",
    "ChatGPTWorkerConfig",
    "ChatSiteConfig",
    "ClaudeWorker",
    "ConversationState",
    "DEEPSEEK_SITE",
    "DEFAULT_PROVIDER",
    "DeepSeekWorker",
    "DisableReason",
    "GEMINI_SITE",
    "GROK_SITE",
    "GeminiWorker",
    "GrokWorker",
    "MISTRAL_SITE",
    "MistralWorker",
    "PERPLEXITY_SITE",
    "PerplexityWorker",
    "QwenWorker",
    "PROVIDER_SITES",
    "ProviderRegistration",
    "ProviderRegistry",
    "QWEN_SITE",
    "WebChatWorker",
    "WorkerConfig",
    "WorkerHealth",
    "WorkerManager",
    "WorkerQuery",
    "WorkerResponse",
    "WorkerState",
    "available_providers",
    "build_registry",
    "build_worker",
    "challenge_prompt",
    "default_disabled_providers",
    "default_enabled_providers",
    "expiry_notice",
    "get_site",
    "login_prompt",
    "render_dashboard",
    "state_glyph",
    "parse_provider_list",
    "verified_providers",
]
