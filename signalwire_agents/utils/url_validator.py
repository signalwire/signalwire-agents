"""
URL validation utility for SSRF protection.

Validates URLs to prevent Server-Side Request Forgery by rejecting
requests to private, internal, or loopback addresses.
"""

import ipaddress
import socket
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """
    Validate a URL for SSRF protection.

    Rejects URLs that point to private, internal, or loopback addresses.
    Only allows http and https schemes.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is safe to request, False otherwise
    """
    try:
        parsed = urlparse(url)

        # Only allow http/https
        if parsed.scheme not in ('http', 'https'):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to IP addresses
        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False

        for addr_info in addr_infos:
            ip = ipaddress.ip_address(addr_info[4][0])

            # Reject private, loopback, reserved, and link-local addresses
            if (ip.is_private or ip.is_loopback or ip.is_reserved
                    or ip.is_link_local or ip.is_multicast):
                return False

        return True

    except (ValueError, TypeError):
        return False
