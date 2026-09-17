"""Sampling remains stable and reservations never exceed the local monthly cap."""
from concurrent.futures import ThreadPoolExecutor


def test_deterministic_sampling_and_concurrent_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("observability:\n  langfuse:\n    sample_rate: 0.25\n    monthly_observation_budget: 12\n")
    from plugins.observability.langfuse.budget import reserve_observation, sampled
    first = [sampled(str(i)) for i in range(100)]
    assert first == [sampled(str(i)) for i in range(100)]
    assert any(first) and not all(first)
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(lambda _: reserve_observation(), range(30)))
    assert sum(admitted) == 12
    assert not reserve_observation()
