"""Outbound inference egress.

Kinoforge does not call model providers directly. It speaks one contract to the Infrelay
gateway (cloud Infrelay or self-host Infrelay-lite), which multiplexes fal, OpenRouter, and
the rest behind it. This package is that single client, kept apart from the inbound HTTP
service layer."""

from kinoforge.inference.infrelay import InfrelayClient, InfrelayError

__all__ = ["InfrelayClient", "InfrelayError"]
