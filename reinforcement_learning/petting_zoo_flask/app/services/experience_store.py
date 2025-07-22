from pymongo import ASCENDING
from app.config.persistance_config import experiences_collection
from app.config.session_state import Experience
from datetime import datetime, timezone

class ExperienceService:
    def __init__(self):
        self.collection = experiences_collection

    def insert_experience(self, env_name: str, experience: Experience):
        if not experience:
            return
        doc = {
            "env_name": env_name,
            "state": experience.state.tolist(),
            "action": int(experience.action),
            "reward": float(experience.reward),
            "next_state": experience.next_state.tolist(),
            "done": bool(experience.done),
            "timestamp": datetime.now(timezone.utc)
        }
        self.collection.insert_one(doc)
        self.enforce_limit(env_name)

    def enforce_limit(self, env_name: str, max_entries: int = 10_000):
        count = self.collection.count_documents({"env_name": env_name})
        if count > max_entries:
            excess = count - max_entries
            old_docs_cursor = self.collection.find(
                {"env_name": env_name},
                sort=[("timestamp", ASCENDING)],
                projection={"_id": 1},
                limit=excess
            )
            old_docs = list(old_docs_cursor)
            ids_to_delete = [doc["_id"] for doc in old_docs]
            self.collection.delete_many({"_id": {"$in": ids_to_delete}})

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
