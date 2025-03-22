# LangGraph Analysis Service

This is a FastAPI backend service that uses LangGraph and LangChain to analyze data using various methodological approaches.

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Local Development

1. Clone the repository
2. Navigate to the `backend/langgraph_service` directory
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```
6. Add your OpenAI API key to the `.env` file
7. Run the server:
   ```
   uvicorn app:app --reload
   ```

The API will be available at `http://localhost:8000`

### Using Docker

1. Build the Docker image:
   ```
   docker build -t langgraph-service .
   ```
2. Run the container:
   ```
   docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here langgraph-service
   ```

## API Endpoints

### POST /run-analysis
Runs analysis on a project using specified agent methodologies.

Request body:
```json
{
  "project_id": "string",
  "user_id": "string",
  "agent_ids": ["string"]
}
```

Response:
```json
{
  "batch_id": "string",
  "insights": [
    {
      "id": "string",
      "text": "string",
      "relevance": 0,
      "methodology": "string",
      "agentId": "string",
      "agentName": "string"
    }
  ],
  "summary": "string"
}
```

## Integration with Frontend

To use this service with the React frontend:

1. Ensure the service is running (either locally or deployed)
2. Set the `USE_LANGGRAPH_BACKEND` environment variable to `true` in your frontend environment
3. The frontend will automatically connect to this service instead of using the simulated responses

## Available Agent Types

- `grounded-theory` - Grounded Theory Agent
- `feminist-theory` - Feminist Theory Agent
- `bias-identification` - Bias Identification Agent
- `critical-analysis` - Critical Analysis Agent
- `phenomenological` - Phenomenological Agent