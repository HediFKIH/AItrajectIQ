from models.metric import Metric
from models.log import Log
from fastapi import HTTPException
from utils.math_utils import cosine_similarity
import numpy as np
import json
import requests
import inspect
from asyncio import to_thread,wait_for,TimeoutError
import warnings
from functools import partial

def ask_openai_if_true(claim,client):
    prompt = f"""
You are a knowledgeable assistant answering based on what you knew up to June 2024.

Given the claim below, assess whether it is:
- "true" if it's clearly correct and you remember it
- "false" if it contradicts known facts
- "unknown" if you're unsure or the information is unavailable

Respond in JSON:
{{
  "decision": "true" | "false" | "unknown",
  "explanation": "short justification"
}}

Claim: "{claim}"
"""
    for i in range(10):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            response_format= { "type": "json_object" }
        )

        try:
            answer=json.loads(response.choices[0].message.content)

            return answer.values()

        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    raise ValueError("the response of 'ask_openai_if_true' isn't being provided in the expected format")


def check_claim_with_context(claim, context, client):
    prompt = f"""
You are a fact-checking assistant.

Given a claim and an external context, classify the relationship as:
- "entail" if the context supports or confirms the claim,
- "contradict" if it clearly refutes the claim,
- "neutral" if the context is unrelated or not sufficient to judge.

Respond ONLY in JSON format like this:
{{
  "verdict": "entail" | "contradict" | "neutral"
}}

Claim: "{claim}"
Context: """ + context
    
    for i in range(10):

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
            response_format= { "type": "json_object" }
        )
        content = response.choices[0].message.content.strip()
        try:
            return json.loads(content)["verdict"].lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    
    raise ValueError("the response isn't being provided in the expected format")


def is_relevant(query,context,client):
    prompt = f""" Is the provided context relevant to the query? Say "yes" if the context contains any information that helps answer the question. Say "no" if there is absolutely nothing relevant in the context (with respect to the query).
Context: '{context}
Query: '{query}' 

Respond ONLY in JSON format like this:
{{
  "decision": "yes" | "no"
}}

"""    
    for i in range(10):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20,
            response_format= { "type": "json_object" }
        )

        decision= response.choices[0].message.content.lower()
        try:
            answer=json.loads(decision)["decision"]
            
            print("context: ", context)
            print("query: " ,query)
            print("222222 ", answer)

            if "yes" in answer:
                return True
            if "no" in answer: 
                return False
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    raise ValueError("the response isn't being provided in the expected format")



async def retrieve_context_from_api(claim: str, host="http://localhost:8000") -> str:
    print('retrieval function entered')
    url = f"{host}/retrieve/"
    payload =  [claim]
    response = await to_thread(partial(requests.post, url, json=payload))

    if response.status_code != 200:
        raise RuntimeError(f"Retrieval failed: {response.status_code} - {response.text}")
    print("retrieval done")

    # Directly parse as a list of lists
    contexts_nested = response.json()

    if not contexts_nested or not isinstance(contexts_nested[0], list):
        return ""

    # Get the list of contexts for the first query
    contexts = contexts_nested[0]
    return "\n\n".join(contexts[:3])  # limit to top 3 if needed


class Metrics:
    def __init__(self,query,answer,metrics_list,embedding_model=None, retrieved_contexts=None, client=None , index=None, category_index=None,match=None,must_contain=None,regex=None):
        self.query=query
        self.response=answer
        self.metrics_list=metrics_list
        self.embedding_model=embedding_model
        self.retrieved_contexts=retrieved_contexts
        self.client=client
        self.index=index
        self.category_index=category_index
        self.match=match
        self.must_contain=must_contain
        self.regex=regex
        self.hallucinations= []

    def break_into_claims(self, text=None):
        if not text:
            text=self.response
        if self.client:
            for i in range(10):
                #print("processing request",i,"...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[               
                        {"role": "system", "content": "From the following paragraph, extract factual claims that can each be independently and unambiguously fact-checked without relying on prior context. Eliminate any use of pronouns or context-dependent references—each claim must be fully self-contained. Ignore personal opinions, greetings, rhetorical phrases, and conversational filler. Present the output as a JSON object with the format: {'1': claim1, '2': claim2, ...}. Each claim should not exceed 100 words. Return only the JSON object."},
                        {"role": "user", "content": "paragraph: " + text}
                    ],
                    temperature=0,
                    response_format= { "type": "json_object" }

                )


                claims_json= response.choices[0].message.content
                try:
                    claims_dict= json.loads(claims_json)
                except json.JSONDecodeError:
                    continue

                claims = list(claims_dict.values())

                return claims
                
            raise ValueError("OpenAI is not extracting information in the expected format")
        
        else:
            raise ValueError("client isn't provided")

    async def general_faithfulness(self):


        claims = self.break_into_claims()
        details = []
        score = 0
        n = 0  # verifiable claim count

        for claim in claims:
            
            n += 1

            decision, explanation = ask_openai_if_true(claim,self.client)

            if decision == "true":
                claim_score = 1.0
                comment = f"LLM judged it as true (pre-knowledge).\nExplanation: {explanation}"
            elif decision == "false":
                claim_score = 0.0
                comment = f"LLM judged it as false (pre-knowledge).\nExplanation: {explanation}"
                details.append({
                    "claim": claim,
                    "comment": comment
                })
                self.hallucinations.append(claim)
            else:  # "unknown"
                # retrieve external context from RAG
                print("waiting for contexts to be retrieved...")
                contexts = await retrieve_context_from_api(claim)
                entailment_result = check_claim_with_context(claim, contexts,self.client)

                if entailment_result == "entail":
                    claim_score = 1.0
                    comment = f"RAG-based validation: Entailed\nContext: {contexts}"
                    details.append({
                        "claim": claim,
                        "comment": comment
                    })
                elif entailment_result == "contradict":
                    claim_score = 0.0
                    comment = f"RAG-based validation: Contradicted\nContext: {contexts}"
                    self.hallucinations.append(claim)
                    details.append({
                        "claim": claim,
                        "comment": comment
                    })

                else:
                    claim_score = 0.5
                    comment = f"RAG-based validation: Neutral or Inconclusive\nContext: {contexts}"

            score += claim_score


        score = score / n if n > 0 else 1.0

        return {
            'value':score,
            'reason': str(details)
            }
    



    def answer_relevance(self,n=3):
        if self.client and self.embedding_model:
            for i in range(10):
                #print("processing request",i,"...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Given the user's answer, generate exactly "+str(n)+" relevant and well-formed questions that could have led to that answer. Return the output in JSON format, where each key is the question number (as a string) and each value is the corresponding question."},
                        {"role": "user", "content": "answer: "+ self.response}
                    ],
                    temperature=0,
                    response_format= { "type": "json_object" }
                )

                
                questions_json= response.choices[0].message.content
                try:
                    questions_dict= json.loads(questions_json)
                except json.JSONDecodeError:
                    continue

                questions = list(questions_dict.values())

                query=self.embedding_model.encode(self.query, normalize_embeddings=True)

                similarities=[]
                for j in range(len(questions)): 
                    question=self.embedding_model.encode(questions[j],normalize_embeddings=True)
                    similarities.append(cosine_similarity(query,question))

                similarities=np.array(similarities)

                res= float(np.sum(similarities)/len(similarities)) 

                return {
                    'value':res,
                    }
    
                
            raise ValueError("the questions aren't being generated in the expected format")
        else:
            raise ValueError(" Answer relevance can't be computed because either client or embedding model isn't provided")


    def completeness(self):

        return {
            'value':0.5,
            }
    def moderation(self):

        return {
            'value':0.5,
            'reason': "any reason"
            } 
    def hallucination(self):

        return {
            'value':self.hallucinations,
            }
    


    def context_precision(self,k=3):
        if not self.retrieved_contexts:
            warnings.warn("Context precision can't be computed because contexts aren't provided")
            return {
                'value':None
                }




        k=min(k,len(self.retrieved_contexts))
        precisions=[]
        Vk=[]

        TP=0
        FP=0
        for i in range(k):

            if is_relevant(self.query,self.retrieved_contexts[i],self.client):
                TP+=1
                Vk.append(1)
            else:
                FP+=1
                Vk.append(0)

            precisions.append(TP/(TP+FP))

        precisions=np.array(precisions)
        Vk=np.array(Vk)

        if TP==0:
            return {
                'value':0
                }
        context_precision= np.sum(precisions*Vk)/TP 
        return {
            'value':float(context_precision)
            }

    async def context_recall(self):
        if not self.retrieved_contexts:
            warnings.warn("Context recall  can't be computed because contexts aren't provided")
            return {
                'value':None
                }
        
        if self.client:
            prompt=f"""
you have enough information to answer the query or do you need more recent data?
query: {self.query}
Say 'yes' if you want additional and more recent contexts, say 'no' if you don't need additional data. 
Respond in JSON:
{{
  "contexts_are_needed": "yes" | "no" 
}}

"""
            
        
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
                response_format= { "type": "json_object" }
            )
            decision = response.choices[0].message.content.strip()

            contexts_are_needed="no"
            
            try:
                dict= json.loads(decision)
                contexts_are_needed=dict["contexts_are_needed"]
            except json.JSONDecodeError:
                print("oops")

            prompt2=self.query
            if contexts_are_needed=="yes":
                gt_contexts= await retrieve_context_from_api(self.query)
                prompt2="contexts: " + gt_contexts+"""\n query: """+prompt2

            response2 = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt2}],
                temperature=0
            )
            gt = response2.choices[0].message.content.strip()

            claims=self.break_into_claims(gt)
            k=0
            print("RECALL \n")
            for claim in claims:
                print("claim",claim,"\n")
                for context in self.retrieved_contexts:
                    print("context:",context,"\n")
                    if check_claim_with_context(context, claim, self.client)=="entail":
                        print("context entails claim")
                        
                        
                        k+=1
                        break
            if len(claims)==0:
                return {
                    'value':0
                    }

            return {
                'value':float(k/len(claims))
                }
        else: 
            raise ValueError("client isn't provided")
        


    def contextual_faithfulness(self):
        if not self.retrieved_contexts:
            warnings.warn("Contextual faithfulness  can't be computed because contexts aren't provided")
            return {
                'value':None
                }


        claims=self.break_into_claims()
        k=0
        details=[]
        
        for claim in claims:
    
            for context in self.retrieved_contexts:
                res=check_claim_with_context(context,claim,self.client)
                if res=="entail":
                    k+=1
                    break

                elif res=="contradict":
                    details.append({
                        "claim": claim,
                        "comment": f"context: {context}"
                    })
                    break
        if len(claims)==0:
            return {
                'value':0
            }

        return {
            'value':k/len(claims),
            'reason':str(details)
            }

    def comet_score(self):

        return {
            'value':0.5
            }

    def summary_completeness(self):

        return {
            'value':0.5
            }
    


    def conciseness(self):

        return {
            'value':0.5
            }
    


    def compression_ratio(self):

        return {
            'value':0.5
            }

    def compilable(self):

        return {
            'value': True
            }
    

    def code_relevance(self):

        return {
            'value':0.5
            }

    def equals(self):

        return {
            'value':False
            }
    
    def contains(self):

        return {
            'value':True
            }
    

    def matches_regex(self):

        return {
            'value':False
            }
    

    def is_json(self):

        return {
            'value': False
            }
    

    def levenshtein(self):

        return {
            'value':0.5
            }





    async def compute_and_save(self,log_id):
        results = []
        for metric_name in self.metrics_list:
            method = getattr(self, metric_name, None)
            if method is None:
                raise ValueError(f"Unknown metric: {metric_name}")
            
            try:
                if inspect.iscoroutinefunction(method):
                    result = await wait_for(method(), timeout=40)
                else:
                    result = await wait_for(to_thread(method), timeout=20)
            except TimeoutError:
                print(f"Timeout while computing {metric_name}")
                continue
        
            results.append(Metric(name=metric_name, **result))
            print(f"{metric_name} is calculated")

        print("metrics are calculated")

        log = await Log.get(log_id)

        print("log is found")
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")


        setattr(log, "metrics", results)

        ("log is updated")
        
        await log.save()
        print("log is saved")
