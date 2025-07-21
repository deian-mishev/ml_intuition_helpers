from datetime import datetime
from pymongo import ASCENDING
from app.config.persistance_config import experiences_collection
from app.config.session_state import Experience

class ExperienceService:
    _instance = None

    def __init__(self):
        self.collection = experiences_collection

    async def insert_experience_batch(self, env_name: str, experiences: list[Experience]):
        if not experiences:
            return
        docs = [{
            "env_name": env_name,
            "state": exp.state.tolist(),
            "action": int(exp.action),
            "reward": float(exp.reward),
            "next_state": exp.next_state.tolist(),
            "done": bool(exp.done),
            "timestamp": datetime.utcnow()
        } for exp in experiences]

        await self.collection.insert_many(docs)
        await self.enforce_limit(env_name)

    async def enforce_limit(self, env_name: str, max_entries: int = 10_000):
        count = await self.collection.count_documents({"env_name": env_name})
        if count > max_entries:
            excess = count - max_entries
            old_docs_cursor = self.collection.find(
                {"env_name": env_name},
                sort=[("timestamp", ASCENDING)],
                projection={"_id": 1},
                limit=excess
            )
            old_docs = await old_docs_cursor.to_list(length=excess)
            ids_to_delete = [doc["_id"] for doc in old_docs]
            await self.collection.delete_many({"_id": {"$in": ids_to_delete}})

    def sample_experiences(self, env_name: str, batch_size: int) -> list[Experience]:
        cursor = self.collection.aggregate([
            {"$match": {"env_name": env_name}},
            {"$sample": {"size": batch_size}}
        ])
        return [Experience(
            state=doc["state"],
            action=doc["action"],
            reward=doc["reward"],
            next_state=doc["next_state"],
            done=doc["done"]
        ) for doc in cursor]


experience_service = ExperienceService()

import asyncio
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

def fire_and_forget_async(coro):
    executor.submit(lambda: asyncio.run(coro))
