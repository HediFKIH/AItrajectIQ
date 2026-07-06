
from beanie import Delete, Document,before_event, PydanticObjectId
from enum import Enum


from models.metric import MetricType, MetricName, METRIC_TYPE_TO_NAMES
from models.log import Log
from models.apikey import APIKey
from pydantic import model_validator
from typing import Literal, Optional, List
from datetime import datetime





class ProjectCategory(str, Enum):
    LLMObserve = "LLMObserve"
    LLMExplain = "LLMExplain"
    LLMInfer = "LLMInfer"


class ProjectTask(str, Enum):
    generic = "generic"
    translation = "translation"
    summarization = "summarization"
    code_generation = "code_generation"
    agent = "agent"


REC_METRIC_TYPES = {
    ProjectTask.generic: {MetricType.generic, MetricType.heuristic, MetricType.summarization_eval, MetricType.translation_eval, MetricType.code_eval, MetricType.RAG_eval},
    ProjectTask.translation: {MetricType.heuristic, MetricType.translation_eval},
    ProjectTask.summarization: {MetricType.heuristic, MetricType.summarization_eval},
    ProjectTask.code_generation: {MetricType.heuristic, MetricType.code_eval},
    ProjectTask.agent: {MetricType.generic, MetricType.heuristic, MetricType.agent_span, MetricType.agent_trace},
}





class Project(Document):
    name:str
    # user_id: PydanticObjectId
    client_id: str
    type: ProjectCategory
    task: Optional[ProjectTask] =None
    enabled_metrics:List[MetricName] = []
    api_key: Optional[APIKey] = None
    created_at: datetime 
    @model_validator(mode='after')
    def check_task(self):
        if self.type == ProjectCategory.LLMObserve:
            if not self.task:
                raise ValueError(f"Task is required for category {self.type}")

            for enabled_metric in self.enabled_metrics:
                metric_is_allowed = False
                for metric_type in REC_METRIC_TYPES[self.task]:
                    if enabled_metric in METRIC_TYPE_TO_NAMES[metric_type]:
                        metric_is_allowed = True
                    if metric_is_allowed:
                        break
                if not metric_is_allowed:
                    raise ValueError(f"the metric {enabled_metric} is not compatible with the project task")

        return self
    class Settings:
        indexes = [
            [
                ("client_id"),
                ("type"),
            ],
        ]

    @before_event(Delete)
    async def delete_logs(self):
        await Log.find(Log.project_id == self.id).delete()
    

 




    