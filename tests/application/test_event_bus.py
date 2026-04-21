import pytest
from application.event_bus import InMemoryEventBus


@pytest.mark.asyncio
async def test_publish_delivers_to_all_subscribers():
    bus = InMemoryEventBus()
    a: list[dict] = []
    b: list[dict] = []

    async def sub_a(e):
        a.append(e)

    async def sub_b(e):
        b.append(e)

    bus.subscribe(sub_a)
    bus.subscribe(sub_b)

    await bus.publish({"x": 1})
    await bus.publish({"x": 2})

    assert a == [{"x": 1}, {"x": 2}]
    assert b == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_subscriber_error_does_not_stop_bus():
    bus = InMemoryEventBus()
    bad_called: list[int] = []
    good: list[dict] = []

    async def bad(e):
        bad_called.append(1)
        raise RuntimeError("boom")

    async def good_sub(e):
        good.append(e)

    bus.subscribe(bad)
    bus.subscribe(good_sub)

    # publish must NOT raise
    await bus.publish({"x": 1})
    await bus.publish({"x": 2})

    assert len(bad_called) == 2
    assert good == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_publish_preserves_order_per_subscriber():
    bus = InMemoryEventBus()
    received: list[int] = []

    async def sub(e):
        received.append(e["i"])

    bus.subscribe(sub)
    for i in range(10):
        await bus.publish({"i": i})

    assert received == list(range(10))
