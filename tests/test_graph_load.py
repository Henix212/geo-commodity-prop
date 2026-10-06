"""Smoke: load metals network from datasets/network."""

from geo_commodity.graph import load_network


def test_load_metals_network():
    net = load_network("metals", validate=False)
    assert net.get("nodes"), "expected nodes in metals network"
    assert net.get("edges") is not None
    assert any(n.get("lat") is not None for n in net["nodes"])
