"""Try IPv4 addresses first. IPv6 connections time out on this network, and Python tries IPv6 first."""
import socket

_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda r: r[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first
