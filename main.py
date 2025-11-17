# main.py
import os
import time
import uuid
import typing as t
from functools import lru_cache
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import jwt

# Embedding + vector store
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

# Async + concurrency
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ---- Configuration (env / defaults) ----
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER_API_KEY = os.environ.get("LLM_API_KEY", "replace-me")
JWT_SECRET = os.environ.get("JWT_SECRET", "supersecret")
JWT_ALGO = "HS256"

# ---- Simple data models ----
@dataclass
class CompanyRecord:
    id: str
    name: str
    domain: str
    industry: t.Optional[str]
    revenue: t.Optional[float]
    intent_topics: t.List[str]
    enriched_at: float

class LeadSummary(BaseModel):
    company_id: str
    company_name: str
    score: float
    summary: str
    recommended_message: str

# ---- Mocked external API clients (replace with real SDKs) ----
class ZoomInfoClient:
    def get_company(self, domain: str) -> dict:
        # placeholder: call ZoomInfo REST API
        # return firmographics like name, industry, revenue
        return {"name": domain.split(".")[0].title(), "domain": domain, "industry": "SaaS", "revenue": 50.0}

class DemandbaseClient:
    def get_intent(self, domain: str) -> dict:
        # placeholder: return intent topics & signal strength
        return {"topics": ["analytics", "data-ingestion"], "score": 0.78}

class PDLClient:
    def get_contacts(self, domain: str) -> list:
        # placeholder: return a list of contact dicts (name, title, email)
        return [
            {"name": "Alice CEO", "title": "CEO", "email": f"alice@{domain}"},
            {"name": "Bob Eng", "title": "Head of Engineering", "email": f"bob@{domain}"}
        ]

# ---- ETL / Normalization ----
def normalize_company(domain: str, zi: ZoomInfoClient, db: DemandbaseClient) -> CompanyRecord:
    zi_data = zi.get_company(domain)
    db_data = db.get_intent(domain)
    record = CompanyRecord(
        id=str(uuid.uuid4()),
        name=zi_data.get("name"),
        domain=domain,
        industry=zi_data.get("industry"),
        revenue=zi_data.get("revenue"),
        intent_topics=db_data.get("topics", []),
        enriched_at=time.time()
    )
    return record

# ---- Embeddings ----
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

# ---- Milvus wrapper ----
class MilvusVectorStore:
    def __init__(self, collection_name="persona_graph"):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection_name = collection_name
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)

    def _create_collection(self):
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="company_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # 384 for MiniLM
        ]
        schema = CollectionSchema(fields, description="PersonaGraph company vectors")
        Collection(self.collection_name, schema=schema)
        # index creation can be added here

    def upsert(self, company: CompanyRecord, vector: t.List[float]):
        # In production you should use partitioning and upsert semantics
        self.collection.insert([[str(uuid.uuid4())], [company.id], [company.name], [vector]])
        self.collection.flush()

    def query(self, vector: t.List[float], top_k=5):
        expr = None
        results = self.collection.search([vector], "vector", param={"metric_type":"L2", "params":{"nprobe":10}}, limit=top_k)
        out = []
        for hits in results:
            for h in hits:
                out.append({"pk": h.id, "score": h.distance, "company_name": h.entity.get("name")})
        return out

# ---- Prompt cache (simple LRU) ----
@lru_cache(maxsize=1024)
def cached_prompt_response(prompt_hash: str):
    # This function is decorated to cache calls by prompt hash.
    # Implementation note: wrap LLM calls and call this with a prompt key.
    return None

# ---- LLM / RAG (placeholder for real LLM) ----
class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In prod, you might connect to OpenAI, Anthropic, Azure, or vLLM endpoint.

    async def generate(self, system: str, prompt: str, max_tokens=256) -> str:
        # Replace this with your async call to LLM provider.
        # For demo, we return a templated response.
        await asyncio.sleep(0.05)  # simulate latency
        return f"[LLM Generated Summary for prompt: {prompt[:100]}...]"

# ---- RAG Generator ----
class RAGGenerator:
    def __init__(self, vector_store: MilvusVectorStore, embedder: Embedder, llm: LLMClient):
        self.vs = vector_store
        self.embedder = embedder
        self.llm = llm

    async def generate_lead_summary(self, company: CompanyRecord) -> LeadSummary:
        # Create a retrieval query: embed company intent topics + name
        contextual = " ".join(company.intent_topics + [company.name])
        vec = self.embedder.embed_texts([contextual])[0]
        neighbors = self.vs.query(vec, top_k=5)
        neighbor_text = "\n".join([n["company_name"] + f" (score:{n['score']})" for n in neighbors])

        prompt = (
            f"Company: {company.name}\n"
            f"Domain: {company.domain}\n"
            f"Industry: {company.industry}\n"
            f"Intent Topics: {', '.join(company.intent_topics)}\n"
            f"Nearby companies (semantic):\n{neighbor_text}\n\n"
            "Write a short qualification summary and a 2-line personalized outreach message."
        )

        # Check prompt cache
        prompt_key = str(hash(prompt))
        cached = cached_prompt_response(prompt_key)
        if cached:
            llm_out = cached
        else:
            llm_out = await self.llm.generate(system="You are a sales assistant.", prompt=prompt)
            # In production, store in a persistent cache (Redis) keyed by prompt hash
        # naive parsing:
        summary = llm_out[:200]
        message = llm_out[200:400] or ("Hi, I noticed you are looking at analytics. Can we talk?")
        score = 0.5 + 0.5 * len(company.intent_topics) / 5.0
        return LeadSummary(company_id=company.id, company_name=company.name, score=score, summary=summary, recommended_message=message)

# ---- FastAPI & Auth (multi-tenant pattern) ----
app = FastAPI(title="PersonaGraph Demo")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Very simple tenant loader from token
def decode_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_tenant(token: str = Depends(oauth2_scheme)):
    payload = decode_jwt(token)
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing tenant")
    return {"tenant_id": tenant_id, "scopes": payload.get("scopes", [])}

# Create singletons (in prod per-tenant instances or namespaces)
embedder = Embedder()
milvus = MilvusVectorStore()
llm = LLMClient(api_key=LLM_PROVIDER_API_KEY)
rag = RAGGenerator(milvus, embedder, llm)

# Threadpool for blocking operations
executor = ThreadPoolExecutor(max_workers=8)

# ---- API endpoints ----
class IngestRequest(BaseModel):
    domain: str

@app.post("/ingest", response_model=LeadSummary)
async def ingest_company(req: IngestRequest, tenant=Depends(get_current_tenant)):
    # For brevity we do everything sync -> run in threadpool
    zi = ZoomInfoClient()
    db = DemandbaseClient()
    pdl = PDLClient()

    loop = asyncio.get_event_loop()
    company = await loop.run_in_executor(executor, normalize_company, req.domain, zi, db)

    # create an embedding and upsert
    contextual = " ".join(company.intent_topics + [company.name])
    vector = await loop.run_in_executor(executor, embedder.embed_texts, [contextual])
    # Milvus upsert
    await loop.run_in_executor(executor, milvus.upsert, company, vector[0])

    # generate a summary via RAG
    lead_summary = await rag.generate_lead_summary(company)
    return lead_summary

@app.get("/health")
def health():
    return {"status": "ok"}

# token endpoint for demo (issue JWT) - in prod use a secure identity provider
class TokenRequest(BaseModel):
    tenant_id: str

@app.post("/token")
def token(req: TokenRequest):
    payload = {"tenant_id": req.tenant_id, "iat": int(time.time())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return {"access_token": token, "token_type": "bearer"}

# ---- If run as script ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
