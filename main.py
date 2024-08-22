import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import json
import boto3
from botocore.config import Config
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
import asyncio
import time
from datetime import datetime
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()

API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI()

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str = Field(default="mistral.mistral-7b-instruct-v0:2", alias="modelId") 
    max_tokens: int = 900
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50
    stream: Optional[bool] = None
    tools: Optional[List[dict]] = None
    options: Optional[dict] = None
    keep_alive: Optional[bool] = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class BedRockClient:
    def __init__(self):
        self.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        # self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        self.aws_region = 'us-east-1'

    def _get_bedrock_client(self, runtime=True):
        retry_config = Config(
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )

        service_name = 'bedrock-runtime' if runtime else 'bedrock'

        bedrock_client = boto3.client(
            service_name=service_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
            config=retry_config
        )

        return bedrock_client

    def list_models(self):
        client = self._get_bedrock_client(runtime=False)
        response = client.list_foundation_models()
        return response['modelSummaries']

    async def invoke_model(self, model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
        client = self._get_bedrock_client(runtime=True)
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })
        

        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        response_body = json.loads(response['body'].read())
        await asyncio.sleep(1)
        return response_body.get('outputs', [{}])[0].get('text', '')
    
    async def invoke_model_stream(self, model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
        client = self._get_bedrock_client(runtime=True)
        
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })

        try:
            response = client.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )

            def process_events():
                for event in response['body']:
                    if event.get('chunk'):
                        chunk_data = json.loads(event['chunk']['bytes'].decode())
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": chunk_data.get('outputs', [{}])[0].get('text', '')
                                },
                                "finish_reason": None
                            }]
                        }

            async for chunk in self._async_generator(process_events()):
                yield chunk

            # After the stream is complete, yield a final chunk with finish_reason "stop"
            yield {
                "choices": [{
                    "delta": {
                        "content": ""
                    },
                    "finish_reason": "stop"
                }]
            }

        except ClientError as e:
            print(f"An error occurred: {e}")
            yield {
                "error": str(e)
            }

    @staticmethod
    async def _async_generator(sync_generator):
        loop = asyncio.get_running_loop()
        for item in sync_generator:
            yield await loop.run_in_executor(None, lambda: item)

# List supported models
@app.get("/models")
async def models(request: Request, api_key: str = Depends(get_api_key)):
    client = BedRockClient()
    
    try:
        supported_models = client.list_models()
        return JSONResponse(content=supported_models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

bedrock_client = BedRockClient()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, data: ChatCompletionRequest, background_tasks: BackgroundTasks):
    checkpoint_start = time.time()
    
    formatted_messages = [
        f"{{'role': '{msg.role}', 'content': '''{msg.content}'''}}"
        for msg in data.messages
    ]
    
    prompt = f"[{', '.join(formatted_messages)}]"

    
    checkpoint_loaded = time.time()

    if not data.messages:
        return JSONResponse(content=ChatResponse(
            model=data.model,
            created_at=time.time(),
            message=Message(role="assistant", content=""),
            done=True,
            done_reason="load"
        ).dict())



    async def generate():
        try:
            full_response = ""
            async for chunk in bedrock_client.invoke_model_stream(
                model_id=data.model,
                prompt=prompt,
                max_tokens=data.max_tokens,
                temperature=data.temperature,
                top_p=data.top_p,
                top_k=data.top_k
            ):
                if chunk and chunk.get('choices'):
                    delta = chunk['choices'][0].get('delta', {}).get('content', '')
                    full_response += delta
                    
                    response_chunk = {
                        "id": f"chat{time.time_ns()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": data.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": delta
                            },
                            "finish_reason": chunk['choices'][0].get('finish_reason')
                        }]
                    }
                    
                    yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    if chunk['choices'][0].get('finish_reason') == "stop":
                        break
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    if data.stream is None or data.stream:
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            response = await bedrock_client.invoke_model(
                model_id=data.model,
                prompt=prompt,
                max_tokens=data.max_tokens,
                temperature=data.temperature,
                top_p=data.top_p,
                top_k=data.top_k
            )
            

            chat_response = ChatResponse(
                id=f"chat{time.time_ns()}",
                created=int(time.time()),
                model=data.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response,
                        },
                        "logprobs": None,  
                        "finish_reason": "stop", 
                    }
                ],
                usage={
                    "prompt_tokens": 0, 
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
            
            return JSONResponse(content=chat_response.dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7002)