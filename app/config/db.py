
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from models.log import Log
from models.project import Project
from models.user import User
from models.record import Record
from models.trace import Trace
from models.span import SPAN_ID_INDEX, Span
from models.apikey import ApiKey
from models.otel_event import OtelEvent
from resources.models.category_mapper import Category
from resources.models.doc_store import Chunk

log = logging.getLogger("aitrajectiq.db")


async def _prepare_span_id_index(database) -> None:
    """Make room for the unique (trace_id, sdk_span_id) index: drop the old non-unique one
    and the duplicates it let in (the most recently written copy is kept)."""
    spans = database["spans"]
    indexes = await spans.index_information()
    if SPAN_ID_INDEX in indexes:
        return
    if "trace_id_1_sdk_span_id_1" in indexes:
        await spans.drop_index("trace_id_1_sdk_span_id_1")
    duplicates = spans.aggregate([
        {"$match": {"sdk_span_id": {"$type": "string"}}},
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": {"t": "$trace_id", "s": "$sdk_span_id"}, "ids": {"$push": "$_id"}, "n": {"$sum": 1}}},
        {"$match": {"n": {"$gt": 1}}},
    ], allowDiskUse=True)
    removed = 0
    async for group in duplicates:
        result = await spans.delete_many({"_id": {"$in": group["ids"][1:]}})
        removed += result.deleted_count
    if removed:
        log.warning("Removed %d duplicate spans before creating the unique span id index", removed)


async def init_db():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    #client = AsyncIOMotorClient("mongodb://localhost:28018")
    #database = client.llm_insight_db2
    database = client.llm_insight_sdk_db
    await _prepare_span_id_index(database)
    await init_beanie(
        database,
        document_models=[
            User,
            Project,
            Log,
            Record,
            Category,
            Chunk,
            Trace,
            Span,
            ApiKey,
            OtelEvent,
        ],
    )
