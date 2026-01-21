import time
from platform.services.cache import InMemoryCache


def test_inmemory_cache_set_get_and_expiry():
    c = InMemoryCache()
    key = "k1"
    v = {"a": 1}
    # set with ttl 1s
    c.set_json(key, v, ttl_s=1)
    assert c.get_json(key) == v
    # wait for expiry
    time.sleep(1.1)
    assert c.get_json(key) is None


def test_inmemory_overwrite_and_invalid_json_not_crash():
    c = InMemoryCache()
    key = "k2"
    c.set_json(key, {"a": 1}, ttl_s=10)
    # overwrite multiple times
    for i in range(3):
        c.set_json(key, {"i": i}, ttl_s=10)
        assert c.get_json(key) == {"i": i}
