"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""URL validation utility to prevent SSRF attacks"""

import ipaddress
import os
import socket
import logging
from urllib.parse import urlparse

logger = logging.getLogger("signalwire.url_validator")

# Private/reserved IP ranges that should be blocked
_BLOCKED_NETWORKS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),  # Link-local / cloud metadata
    ipaddress.ip_network('0.0.0.0/8'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),  # IPv6 private
    ipaddress.ip_network('fe80::/10'),  # IPv6 link-local
]


def validate_url(url: str, allow_private: bool = False) -> bool:
    """
    Validate that a URL is safe to fetch (not pointing to private/internal resources).

    Args:
        url: The URL to validate
        allow_private: If True, allow private IP ranges (default: False)

    Returns:
        True if the URL is safe to fetch, False otherwise
    """
    try:
        parsed = urlparse(url)

        # Require http or https scheme
        if parsed.scheme not in ('http', 'https'):
            logger.warning("URL rejected: invalid scheme %s", parsed.scheme)
            return False

        # Must have a hostname
        hostname = parsed.hostname
        if not hostname:
            logger.warning("URL rejected: no hostname")
            return False

        if allow_private or os.getenv('SWML_ALLOW_PRIVATE_URLS', '').lower() in ('1', 'true', 'yes'):
            return True

        # Resolve hostname to IP addresses
        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            logger.warning("URL rejected: could not resolve hostname %s", hostname)
            return False

        # Check all resolved IPs against blocked ranges
        for addr_info in addr_infos:
            ip_str = addr_info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
                for network in _BLOCKED_NETWORKS:
                    if ip in network:
                        logger.warning(
                            "URL rejected: %s resolves to blocked IP %s (in %s)",
                            hostname, ip_str, network
                        )
                        return False
            except ValueError:
                continue

        return True

    except Exception as e:
        logger.warning("URL validation error: %s", e)
        return False
