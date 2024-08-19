# Drop-in Replacement for OpenAI Chat Completion

## Overview

This is a FastAPI-based application that serves as a drop-in replacement for OpenAI's chat API, leveraging AWS Bedrock. This project allows developers to seamlessly switch from OpenAI to AWS Bedrock without changing their existing integration code.



## Prerequisites
- AWS account with Bedrock access
- AWS credentials (Access Key ID and Secret Access Key)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/9prodhi/bedrock-openai-proxy.git
   cd bedrock-openai-proxy
   ```

2. Create a virtual environment and activate it:
   ```
   conda create -n chat-api python=3.12
   conda activate chat-api
   ```

3. Install the required dependencies:
   ```
   pip install fastapi uvicorn python-dotenv boto3 
   ```

4. Create a `.env` file in the project root and add your configuration:
   ```
   AWS_ACCESS_KEY_ID=999
   AWS_SECRET_ACCESS_KEY=99999
   AWS_REGION=us-east-1  # Optional, defaults to us-east-1
   ```

## Usage

1. Start the server:
   ```
   python main.py
   ```

2. The API will be available at `http://localhost:7002` 

3. Use the following endpoints:

   - GET `/`: Returns the index.html file
   - GET `/models`: Lists supported models
   - POST `v1/chat/completions`: Main endpoint for chat completions (OpenAI-compatible)


4. To make a request to the `v1/chat/completions` endpoint, use the same format as OpenAI's chat API:

   ```json
   {
     "messages": [
       {"role": "system", "content": "You are a helpful AI."},
       {"role": "user", "content": "What is Encoder?"}
     ],
     "model": "mistral.mistral-7b-instruct-v0:2",
     "max_tokens": 800,
     "temperature": 0.1,
     "top_p": 0.9,
     "stream": true
   }
   ```

