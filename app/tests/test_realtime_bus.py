import asyncio

from realtime.bus import EventBus


def run(coro):
    return asyncio.run(coro)


def test_subscriber_receives_published_event():
    async def scenario():
        bus = EventBus()
        with bus.subscribe("p1") as sub:
            assert bus.publish("p1", "trace.started", {"trace": {"id": "t1"}}) == 1
            return await sub.next(timeout=1)

    event = run(scenario())
    assert event["type"] == "trace.started"
    assert event["data"] == {"trace": {"id": "t1"}}
    assert isinstance(event["ts"], float)


def test_events_are_isolated_per_project():
    async def scenario():
        bus = EventBus()
        with bus.subscribe("p1") as sub1, bus.subscribe("p2") as sub2:
            assert bus.publish("p2", "span.created", {"trace_id": "t", "count": 1}) == 1
            return await sub1.next(timeout=0.05), await sub2.next(timeout=1)

    first, second = run(scenario())
    assert first is None
    assert second["type"] == "span.created"


def test_project_ids_are_compared_as_strings():
    class Oid:
        def __str__(self):
            return "abc"

    async def scenario():
        bus = EventBus()
        with bus.subscribe("abc") as sub:
            bus.publish(Oid(), "metrics.updated")
            return await sub.next(timeout=1)

    event = run(scenario())
    assert event["type"] == "metrics.updated"
    assert event["data"] == {}


def test_publish_without_subscribers_delivers_nothing():
    assert EventBus().publish("nobody", "trace.started", {}) == 0


def test_slow_subscriber_drops_events_and_counts_them():
    async def scenario():
        bus = EventBus(queue_size=2)
        with bus.subscribe("p") as slow, bus.subscribe("p") as fast:
            delivered = [bus.publish("p", "span.created", {"n": i}) for i in range(2)]
            # Drain the fast subscriber so only the slow one overflows.
            await fast.next(timeout=1)
            await fast.next(timeout=1)
            delivered.append(bus.publish("p", "span.created", {"n": 2}))
            first = await slow.next(timeout=1)
            return delivered, first, slow.take_dropped(), slow.take_dropped()

    delivered, first, dropped, dropped_again = run(scenario())
    assert delivered == [2, 2, 1]
    assert first["data"] == {"n": 0}
    assert dropped == 1
    assert dropped_again == 0


def test_subscription_is_removed_on_exit():
    bus = EventBus()
    with bus.subscribe("p"):
        with bus.subscribe("p"):
            assert bus.subscriber_count("p") == 2
        assert bus.subscriber_count("p") == 1
    assert bus.subscriber_count("p") == 0
    assert bus.subscriber_count() == 0
    assert bus.publish("p", "trace.started") == 0


def test_next_times_out_with_none():
    async def scenario():
        with EventBus().subscribe("p") as sub:
            return await sub.next(timeout=0.01)

    assert run(scenario()) is None
