"""
Provider registry with explicit enable and disable semantics.

Which providers participate is configuration, never code. A provider that
proves unusable is disabled rather than deleted, so its selectors, timings,
and capabilities stay in the repository and re-enabling it is a one-line
change to the environment.

Resolution rules, in precedence order:

1. A provider named in the disabled list is disabled. Nothing overrides
   this, including naming it in the enabled list — an explicit refusal is
   never quietly reversed by an explicit request.
2. A provider named in the enabled list and not disabled is enabled.
3. With no enabled list configured, every provider whose site config is
   ``enabled_by_default`` is enabled.
4. Unknown names are reported and ignored, so a typo degrades to the
   providers that were understood rather than failing startup.

Ordering follows the enabled list when one is given, otherwise registry
order, so operators control which provider opens its tab first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from src.config import Settings, get_settings
from src.exceptions import WorkerError
from src.logger import get_logger

from .chat_site import ChatSiteConfig


class DisableReason(str, Enum):
    """
    Why a provider is not participating.

    Attributes:
        CONFIGURED: Named in the disabled list.
        DEFAULT: Ships disabled because it is known to be unusable.
        NOT_SELECTED: Not named in an explicit enabled list.
    """

    CONFIGURED = "disabled_by_configuration"
    DEFAULT = "disabled_by_default"
    NOT_SELECTED = "not_in_enabled_list"


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """
    Resolved participation decision for one provider.

    Attributes:
        name: Provider name.
        site: Provider site description.
        enabled: Whether the provider participates.
        reason: Why it was disabled, when it is not enabled.
        detail: Human-readable explanation, taken from the site config
            when the provider ships disabled.
    """

    name: str
    site: ChatSiteConfig
    enabled: bool
    reason: Optional[DisableReason] = None
    detail: str = ""

    @property
    def display_name(self) -> str:
        """
        Return the human-readable provider name.

        Returns:
            Display name.
        """
        return self.site.display_name

    def describe(self) -> str:
        """
        Render a single-line summary for startup logs.

        Returns:
            Human-readable description.
        """
        if self.enabled:
            tag = "" if self.site.verified else " (unverified selectors)"
            return f"{self.display_name}{tag}"

        reason = self.reason.value if self.reason else "disabled"
        suffix = f" — {self.detail}" if self.detail else ""
        return f"{self.display_name} [{reason}]{suffix}"


def _parse_names(raw: str) -> list[str]:
    """
    Parse a comma-separated provider list.

    Args:
        raw: Raw configuration value.

    Returns:
        Lower-cased names in order, without duplicates or blanks.
    """
    names: list[str] = []
    for candidate in (raw or "").split(","):
        name = candidate.strip().lower()
        if name and name not in names:
            names.append(name)
    return names


class ProviderRegistry:
    """
    Resolves which providers participate, from configuration.

    The registry is the single authority on participation. Every consumer —
    session manager, health monitor, recovery, orchestrator — asks it
    rather than holding its own list, so a disabled provider cannot leak
    into one code path while being absent from another.
    """

    def __init__(
        self,
        sites: dict[str, ChatSiteConfig],
        enabled: Optional[Iterable[str]] = None,
        disabled: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Initialize the registry.

        Args:
            sites: All known provider site descriptions, keyed by name.
            enabled: Explicitly enabled provider names. When None, every
                provider that is enabled by default participates.
            disabled: Explicitly disabled provider names, which always win.
        """
        self._sites = dict(sites)
        self._logger = get_logger(__name__)

        self._requested = list(enabled) if enabled is not None else None
        self._refused = list(disabled or [])
        self._registrations = self._resolve()

    @classmethod
    def from_settings(
        cls,
        sites: dict[str, ChatSiteConfig],
        settings: Optional[Settings] = None,
    ) -> "ProviderRegistry":
        """
        Build a registry from application settings.

        Args:
            sites: All known provider site descriptions.
            settings: Optional application settings.

        Returns:
            Configured registry.
        """
        resolved = settings or get_settings()
        requested = _parse_names(resolved.enabled_providers)
        return cls(
            sites=sites,
            enabled=requested or None,
            disabled=_parse_names(resolved.disabled_providers),
        )

    def _resolve(self) -> dict[str, ProviderRegistration]:
        """
        Apply the resolution rules to every known provider.

        Returns:
            Registration per provider, ordered with enabled providers in
            configured order first.
        """
        refused = set(self._refused)
        unknown = [
            name
            for name in (self._requested or []) + self._refused
            if name not in self._sites
        ]
        if unknown:
            self._logger.warning(
                "Ignoring unknown provider name(s) in configuration: %s. "
                "Known providers: %s",
                ", ".join(sorted(set(unknown))),
                ", ".join(sorted(self._sites)),
            )

        conflicting = [
            name for name in (self._requested or []) if name in refused
        ]
        if conflicting:
            self._logger.warning(
                "Provider(s) %s appear in both the enabled and disabled "
                "lists; the disabled list wins",
                ", ".join(sorted(conflicting)),
            )

        ordering = [
            name for name in (self._requested or []) if name in self._sites
        ]
        ordering += [name for name in self._sites if name not in ordering]

        registrations: dict[str, ProviderRegistration] = {}
        for name in ordering:
            site = self._sites[name]

            if name in refused:
                registrations[name] = ProviderRegistration(
                    name=name,
                    site=site,
                    enabled=False,
                    reason=DisableReason.CONFIGURED,
                    detail=site.disabled_reason,
                )
                continue

            if self._requested is not None:
                if name in self._requested:
                    registrations[name] = ProviderRegistration(
                        name=name, site=site, enabled=True
                    )
                else:
                    registrations[name] = ProviderRegistration(
                        name=name,
                        site=site,
                        enabled=False,
                        reason=DisableReason.NOT_SELECTED,
                        detail=site.disabled_reason,
                    )
                continue

            if site.enabled_by_default:
                registrations[name] = ProviderRegistration(
                    name=name, site=site, enabled=True
                )
            else:
                registrations[name] = ProviderRegistration(
                    name=name,
                    site=site,
                    enabled=False,
                    reason=DisableReason.DEFAULT,
                    detail=site.disabled_reason,
                )

        return registrations

    @property
    def registrations(self) -> tuple[ProviderRegistration, ...]:
        """
        Return every registration, enabled providers first.

        Returns:
            All registrations in resolution order.
        """
        return tuple(self._registrations.values())

    def enabled_names(self) -> list[str]:
        """
        Return the providers that participate, in configured order.

        Returns:
            Enabled provider names.
        """
        return [
            registration.name
            for registration in self._registrations.values()
            if registration.enabled
        ]

    def disabled_names(self) -> list[str]:
        """
        Return the providers that do not participate.

        Returns:
            Disabled provider names.
        """
        return [
            registration.name
            for registration in self._registrations.values()
            if not registration.enabled
        ]

    def is_enabled(self, name: str) -> bool:
        """
        Report whether a provider participates.

        Unknown providers are reported as disabled rather than raising, so
        a caller filtering a list never has to guard the lookup.

        Args:
            name: Provider name.

        Returns:
            True when the provider is enabled.
        """
        registration = self._registrations.get(name.strip().lower())
        return registration is not None and registration.enabled

    def require_enabled(self, name: str) -> ProviderRegistration:
        """
        Return a provider's registration, refusing disabled providers.

        Args:
            name: Provider name.

        Returns:
            Registration for an enabled provider.

        Raises:
            WorkerError: If the provider is unknown or disabled.
        """
        key = name.strip().lower()
        registration = self._registrations.get(key)

        if registration is None:
            raise WorkerError(
                f"Unknown provider '{name}'. Known providers: "
                f"{', '.join(sorted(self._sites))}",
                code="PROVIDER_UNKNOWN",
            )

        if not registration.enabled:
            reason = (
                registration.reason.value
                if registration.reason
                else "disabled"
            )
            raise WorkerError(
                f"Provider '{name}' is disabled ({reason}). Add it to "
                f"ENABLED_PROVIDERS to use it.",
                code="PROVIDER_DISABLED",
            )
        return registration

    def filter_enabled(self, names: Iterable[str]) -> list[str]:
        """
        Drop disabled and unknown providers from a candidate list.

        The single choke point every consumer uses, so a disabled provider
        cannot reach dispatch, monitoring, or recovery by any route.

        Args:
            names: Candidate provider names.

        Returns:
            Enabled names, order preserved, duplicates removed.
        """
        kept: list[str] = []
        for name in names:
            key = name.strip().lower()
            if self.is_enabled(key) and key not in kept:
                kept.append(key)
        return kept

    def startup_summary(self) -> str:
        """
        Render the enabled and disabled provider lists for startup logs.

        Returns:
            Multi-line summary.
        """
        lines = ["Enabled Providers:"]
        enabled = [r for r in self._registrations.values() if r.enabled]
        if enabled:
            lines.extend(f"  {r.describe()}" for r in enabled)
        else:
            lines.append("  (none)")

        disabled = [
            r for r in self._registrations.values() if not r.enabled
        ]
        lines.append("Disabled Providers:")
        if disabled:
            lines.extend(f"  {r.describe()}" for r in disabled)
        else:
            lines.append("  (none)")

        return "\n".join(lines)

    def log_summary(self) -> None:
        """Write the startup summary to the log, one line per provider."""
        for line in self.startup_summary().splitlines():
            self._logger.info("%s", line)


__all__ = [
    "DisableReason",
    "ProviderRegistration",
    "ProviderRegistry",
]
