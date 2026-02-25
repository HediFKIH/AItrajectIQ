from typing import List
from models.metric import Metric
from models.project import Project
from fastapi import Request, HTTPException
from .models import ChatCompletionRequest, Message, RegisterLogRequest,ResponseData
from models.log import Log
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from llm_guard.output_scanners import Deanonymize
from llm_guard.vault import Vault
from openai import AsyncOpenAI
from llm_guard.input_scanners.prompt_injection import PromptInjection, MatchType
from beanie import PydanticObjectId
import time
from uuid import uuid4




def extract_credentials(request:Request,body):
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        api_key = auth.split(" ", 1)[1]
    else:
        api_key = None
    
    #body = await request.json()
    #body = ChatCompletionRequest(**body)
    project_id= body.project_id

    return project_id,api_key

async def auth_verify(project_id,auth):

    project = await Project.get(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    
    if auth!=project.api_key.key:
        print(auth)
        print(project.api_key.key)
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")
    return project
 
    



def extract_fields(body:ChatCompletionRequest):

    messages=body.messages
    model=None
    temperature=None
    max_tokens=None
    hidden_names=None
    contexts=None
    if body.model is not None:
        model=body.model
    
    if body.max_tokens is not None:
        max_tokens=body.max_tokens
    if body.temperature is not None:
        temperature=body.temperature

    if body.hidden_names is not None:
        hidden_names=body.hidden_names
    if body.contexts is not None:
        contexts=body.contexts

    provider_endpoint= body.provider_endpoint
    provider_key=body.provider_key

    return messages,model,temperature,max_tokens, provider_endpoint,provider_key,hidden_names,contexts






def check_for_block(prompt,prompt_injection_scanner):
    _, is_valid_pi, risk_score_pi = prompt_injection_scanner.scan(prompt)

    if not is_valid_pi:
        return True, f"Prompt Injection detected with risk score {risk_score_pi}"

    return False,""


def anonymize(input: str,scanner):
    sanitized_prompt, is_valid, risk_score = scanner.scan(input)
    contains_PII=not(is_valid)
    return sanitized_prompt, contains_PII,risk_score

def update_history(sanitized_prompt,history: list[Message]):
    for i in range(len(history)-1,-1,-1):
        if history[i].role == "user":
            history[i].content=sanitized_prompt
            break
        



#def save_log(fields,project_id):
async def create_log(log_data: RegisterLogRequest): #log_data contains everything, including the project id
    log = Log(**log_data.model_dump())
    await log.insert()
    return log.id

#did not delete this cuz ill need it when updating the log after metric calculation

# async def add_request_analysis(log_id: str, data: RequestAnalysisData):
#     log = await Log.get(log_id)
#     if not log:
#         raise HTTPException(status_code=404, detail="Log not found")

#     update_data = data.model_dump(exclude_unset=True)  # only fields sent in request
#     for field, value in update_data.items():
#         setattr(log, field, value)
    
#     await log.save()
#     return log

async def forward_request(provider_url,model, history, max_token=None, temperature=None, api_key=None):
    #communication with provider logic
    client = AsyncOpenAI(
        base_url=provider_url,
        api_key=api_key,

    )
    args={
        "messages":history
    }
    if model:
        args["model"]=model
    if max_token:
        args["max_tokens"]=max_token
    if temperature:
        args["temperature"]=temperature


    response= await client.chat.completions.create(**args)

    return response




async def add_response_data(log_id: PydanticObjectId, data: ResponseData):
    log = await Log.get(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    update_data = data.model_dump(exclude_unset=True)  # only fields sent in request
    for field, value in update_data.items():
        setattr(log, field, value)
    
    await log.save()
    

def deanonymize(sanitized_prompt: str, answer: str,vault: Vault):
    scanner = Deanonymize(vault)
    sanitized_model_output, _,_ = scanner.scan(sanitized_prompt, answer)
    
    return sanitized_model_output

def format_openai_style_response(content: str, model: str, prompt_tokens: int, completion_tokens: int):
    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",  # Random ID like OpenAI's #fixxxxxxxxxxxxxxxxxxxxxxx obviously get the id from the response and other fields too
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }






#to define later after finishing metrics 

def calculate_metrics(enabled_metrics):
    return list[Metric]

def save_metrics(metrics):
    return None


    