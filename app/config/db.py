
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from models.log import Log
from models.project import Project
from models.user import User
from models.record import Record
from models.trace import Trace
from models.span import Span


from resources.models.category_mapper import Category
from resources.models.doc_store import Chunk


async def init_db():
    # client = AsyncIOMotorClient("mongodb://localhost:27017")
    client = AsyncIOMotorClient("mongodb://localhost:28018")
    #database = client.llm_insight_db2
    database = client.llm_insight_sdk_db
    await init_beanie(database, document_models=[User, Project, Log, Record, Category, Chunk, Trace, Span])

