"""Redpanda (Kafka-API) event bus helpers.

Topics are the plane handoffs: frame.ready, labels.ready, object.gated. Producers and consumers
are async (aiokafka). The bus is fire-and-forward; Postgres remains the system of record, so a
dropped event never loses data, it only delays a downstream consumer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

TOPIC_FRAME_READY = "frame.ready"
TOPIC_LABELS_READY = "labels.ready"
TOPIC_OBJECT_GATED = "object.gated"
TOPIC_PII_AUDIT = "pii.audit"
TOPIC_IMPORT_REQUESTED = "import.requested"  # cloud-seam worker topic (consumer not built yet)
TOPIC_SCENE_READY = "scene.ready"  # frame scene tags written (Data Intelligence Layer M1.3)

ALL_TOPICS = [TOPIC_FRAME_READY, TOPIC_LABELS_READY, TOPIC_OBJECT_GATED, TOPIC_PII_AUDIT, TOPIC_SCENE_READY]


class EventBus:
    def __init__(self, brokers: str | None = None) -> None:
        self.brokers = brokers or get_settings().redpanda.brokers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                enable_idempotence=True,
                acks="all",
            )
            await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, value: dict, key: str | None = None) -> None:
        if self._producer is None:
            await self.start()
        assert self._producer is not None
        await self._producer.send_and_wait(topic, value=value, key=key)
        log.debug("bus.publish", topic=topic, key=key)

    async def emit(self, topic: str, value: dict, key: str | None = None) -> None:
        """Buffered, non-blocking publish: queues the record (producer batches and flushes on stop)
        without waiting for the broker ack. For high-volume per-object events where Postgres remains
        the system of record, so a delayed ack never loses data."""
        if self._producer is None:
            await self.start()
        assert self._producer is not None
        await self._producer.send(topic, value=value, key=key)

    async def consume(
        self, topics: list[str], group_id: str, from_beginning: bool = False
    ) -> AsyncIterator[dict]:
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.brokers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest" if from_beginning else "latest",
            enable_auto_commit=True,
        )
        await consumer.start()
        try:
            async for msg in consumer:
                yield {"topic": msg.topic, "key": msg.key, "value": msg.value}
        finally:
            await consumer.stop()


async def __aenter_bus() -> EventBus:  # pragma: no cover - convenience only
    bus = EventBus()
    await bus.start()
    return bus
