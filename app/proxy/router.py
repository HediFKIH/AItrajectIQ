from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# from forwarder import forward_request
# from auth import resolve_provider
#from .models import ChatCompletionRequest
from .processing import *
from llm_guard.vault import Vault
from utils.time import now_utc_plus_one
from pydantic import ValidationError
import time
from .metrics import Metrics

import asyncio




router = APIRouter()




@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start = time.monotonic()
    try:
        body = await request.json()
        parsed_body = ChatCompletionRequest(**body)

    except ValidationError as ve:
        return JSONResponse(
            status_code=422,
            content={"detail": ve.errors()}
        )
    

    pi_scanner=request.app.state.prompt_injection_scanner
    open_client=request.app.state.openai_client
    embedding_model=request.app.state.embedding_model
    translation_eval_model=request.app.state.translation_eval_model
    project_id, auth= extract_credentials(request, parsed_body)
    verified_project= await auth_verify(project_id, auth)
    metrics_list = verified_project.enabled_metrics
    

    messages,model,temperature, max_tokens,provider_endpoint,provider_key,hidden_names,contexts=extract_fields(parsed_body)


    
        

    def get_last_user_prompt(messages: list[Message]):
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""
    prompt=get_last_user_prompt(messages)
    if contexts: 
        context = "\n".join(contexts)
        prompt="Context: " + context + "\n" + prompt



    blocked,block_reason=check_for_block(prompt,pi_scanner)

    

    vault=Vault()

    scanner = Anonymize(vault,hidden_names=hidden_names,recognizer_conf=BERT_LARGE_NER_CONF, language="en")

    contains_PII=False
    for i in range(len(messages)):
        message=messages[i].content
        sanitized_message,contains_PII,_=anonymize(message,scanner)
        messages[i].content=sanitized_message

    prompt=get_last_user_prompt(messages)
    

    if blocked:
        log_data = RegisterLogRequest(
            project_id=project_id,
            prompt=prompt,
            history=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            retrieved_contexts=contexts,
            timestamp=now_utc_plus_one(),
            blocked=True,
            blocked_reason=block_reason,
            contains_pii=contains_PII,
            anonymized=contains_PII
        )
        await create_log(log_data)
        return JSONResponse(status_code=403, content={"error": block_reason})


    log_data=RegisterLogRequest(
        project_id=project_id,
        prompt= prompt,
        history=messages,
        temperature= temperature,
        max_tokens=max_tokens,
        retrieved_contexts=contexts,
        timestamp = now_utc_plus_one(),
        blocked= blocked,
        blocked_reason= block_reason,
        contains_pii=contains_PII,
        anonymized=contains_PII
    )
    log_id = await create_log(log_data)
    assert log_id is not None, "Insert failed — log ID is missing"
    response= await forward_request(provider_url=provider_endpoint,model= model, history=messages,max_token=max_tokens, temperature=temperature,api_key=provider_key )


    anonymized_response=response.choices[0].message.content

    end = time.monotonic()
    latency = int(end - start)
    data= ResponseData(
        Output= anonymized_response,
        Model=response.model,
        latency=latency,
        n_tokens=response.usage.completion_tokens
    )

    await add_response_data(log_id, data)

    deanonymized_response= deanonymize(prompt,anonymized_response,vault=vault)

    formatted_response=format_openai_style_response(deanonymized_response, model= response.model, prompt_tokens= response.usage.prompt_tokens, completion_tokens= response.usage.completion_tokens)

    metrics_obj=Metrics(prompt, anonymized_response, metrics_list, retrieved_contexts=contexts, client=open_client,embedding_model=embedding_model, translation_eval_model=translation_eval_model)

    asyncio.create_task(metrics_obj.compute_and_save(log_id))


    return JSONResponse(content=formatted_response)