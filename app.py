from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langgraph.graph import StateGraph
from langchain.schema import HumanMessage, AIMessage

app = FastAPI(title="LangGraph Analysis Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class AnalysisRequest(BaseModel):
    project_id: str
    user_id: str
    agent_ids: List[str]

# Response models
class AnalysisResponse(BaseModel):
    batch_id: str
    insights: List[dict]
    summary: str

# Mock agents for demonstration
AGENT_METHODS = {
    "grounded-theory": "Grounded Theory Agent",
    "feminist-theory": "Feminist Theory Agent",
    "bias-identification": "Bias Identification Agent",
    "critical-analysis": "Critical Analysis Agent",
    "phenomenological": "Phenomenological Agent"
}

def create_agent_for_method(agent_id: str):
    """Create a specific agent based on methodology"""
    # This would be more sophisticated in a real implementation
    llm = ChatOpenAI(temperature=0.7)
    
    @tool
    def analyze_data(query: str) -> str:
        """Analyze data using the specific methodology"""
        method_name = AGENT_METHODS.get(agent_id, "Unknown Method")
        return f"Analysis from {method_name}: insights about {query}"
    
    memory = ConversationBufferMemory()
    
    # Create a simple state graph for the agent
    class AgentState:
        def __init__(self):
            self.messages = []
            self.next = None
    
    def agent_node(state):
        messages = state.messages
        response = llm.invoke(messages[-1].content if messages else "No input provided")
        return {"messages": messages + [AIMessage(content=response.content)], "next": "output"}
    
    def output_node(state):
        return state
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("output", output_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "output")
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph

@app.get("/")
async def root():
    return {"message": "LangGraph Analysis Service is running"}

@app.post("/run-analysis", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    try:
        # Generate a unique batch ID
        batch_id = str(uuid.uuid4())
        
        # Validate agent_ids
        valid_agents = []
        for agent_id in request.agent_ids:
            if agent_id in AGENT_METHODS:
                valid_agents.append(agent_id)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_id}")
        
        if not valid_agents:
            raise HTTPException(status_code=400, detail="No valid agents specified")
        
        # Process each agent (in a real system, this would be parallelized or queued)
        insights = []
        for agent_id in valid_agents:
            # Create and run the agent
            agent = create_agent_for_method(agent_id)
            
            # In a real implementation, you would process actual data
            # For demo purposes, we're generating mock insights
            agent_name = AGENT_METHODS[agent_id]
            
            # Create mock insights based on agent type
            if agent_id == "grounded-theory":
                insights.append({
                    "id": f"lg-{uuid.uuid4()}",
                    "text": "Users frequently mentioned difficulties with the navigation interface, particularly on mobile devices.",
                    "relevance": 92,
                    "methodology": "Grounded Theory",
                    "agentId": agent_id,
                    "agentName": agent_name
                })
            elif agent_id == "feminist-theory":
                insights.append({
                    "id": f"lg-{uuid.uuid4()}",
                    "text": "Significant gender disparity in reporting technical issues, with women more likely to articulate specific interface problems.",
                    "relevance": 85,
                    "methodology": "Feminist Theory",
                    "agentId": agent_id,
                    "agentName": agent_name
                })
            elif agent_id == "bias-identification":
                insights.append({
                    "id": f"lg-{uuid.uuid4()}",
                    "text": "Documentation contains technical jargon that creates barriers for non-expert users.",
                    "relevance": 78,
                    "methodology": "Critical Analysis",
                    "agentId": agent_id,
                    "agentName": agent_name
                })
            elif agent_id == "critical-analysis":
                insights.append({
                    "id": f"lg-{uuid.uuid4()}",
                    "text": "The current product narrative implies a universal user experience while ignoring cultural context variation.",
                    "relevance": 82,
                    "methodology": "Critical Analysis",
                    "agentId": agent_id,
                    "agentName": agent_name
                })
            elif agent_id == "phenomenological":
                insights.append({
                    "id": f"lg-{uuid.uuid4()}",
                    "text": "Users express feelings of frustration during onboarding, followed by confidence after completing initial tasks.",
                    "relevance": 88,
                    "methodology": "Phenomenological Analysis",
                    "agentId": agent_id,
                    "agentName": agent_name
                })
        
        # Generate a summary based on all insights
        summary = "Analysis reveals significant usability challenges with the navigation interface, particularly on mobile devices. Gender disparities exist in how technical issues are reported, with women providing more specific details about interface problems. The documentation uses technical jargon that creates barriers for non-expert users."
        
        return AnalysisResponse(
            batch_id=batch_id,
            insights=insights,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)