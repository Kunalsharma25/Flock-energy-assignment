import src.config as C

def test_config_import():
    assert C.SEED == 42
    assert C.APPLIANCE_SPIKE_W == 4000
    assert len(C.WM_CANDIDATES) == 20
