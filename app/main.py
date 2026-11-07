import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent import Agent, ToolBudgetExceeded
from app.config import PACKAGE_DIR, Settings
from app.database import OrderStore, initialize_database
from app.embeddings import LocalChineseEncoder
from app.llm import ChatCompletionClient, ProviderError
from app.retrieval import DocumentIndex
from app.schemas import ChatRequest, ChatResponse
from app.tools import ToolRegistry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_path = settings.data_dir / "orders.db"
        initialize_database(db_path)
        encoder = (
            LocalChineseEncoder(settings.data_dir / "models")
            if settings.retrieval_mode != "bm25"
            else None
        )
        index = DocumentIndex(
            PACKAGE_DIR / "sample_data/docs",
            mode=settings.retrieval_mode,
            encoder=encoder,
            cache_dir=settings.data_dir / "vectors",
            candidate_k=settings.retrieval_candidate_k,
            rrf_k=settings.retrieval_rrf_k,
        )
        registry = ToolRegistry(index, OrderStore(db_path))
        async with httpx.AsyncClient(follow_redirects=False) as client:
            app.state.agent = Agent(settings, registry, ChatCompletionClient(settings, client))
            app.state.index = index
            # A local MVP capacity guard: reject extra work instead of unbounded queuing.
            app.state.active_requests = 0
            yield

    app = FastAPI(title="Enterprise AI Agent", version="0.2.0", lifespan=lifespan, redoc_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        # FastAPI's Swagger page uses a CDN and an inline initialization script.
        if request.url.path in {"/docs", "/docs/oauth2-redirect"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'"
            )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": settings.agent_mode}

    @app.get("/api/config")
    async def config():
        return {
            "mode": settings.agent_mode,
            "model": settings.openai_model if settings.agent_mode == "live" else None,
            "ready": settings.agent_mode == "demo"
            or bool(settings.openai_api_key.get_secret_value()),
            "documents": len(app.state.index.documents),
            "chunks": len(app.state.index.chunks),
            "retrieval_mode": app.state.index.mode,
            "embedding_model": LocalChineseEncoder.MODEL if app.state.index.dense else None,
            "vector_cache_hit": (
                app.state.index.dense.cache_hit if app.state.index.dense else None
            ),
            "tools": ["search_documents", "query_orders"],
        }

    @app.get("/api/documents")
    async def documents():
        return [{"id": doc["id"], "title": doc["title"]} for doc in app.state.index.documents]

    @app.get("/api/documents/{document_id}")
    async def document(document_id: str):
        for doc in app.state.index.documents:
            if doc["id"] == document_id:
                return doc
        raise HTTPException(404, "文档不存在")

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest):
        if settings.agent_mode == "live" and not settings.openai_api_key.get_secret_value():
            raise HTTPException(503, "live 模式缺少 OPENAI_API_KEY，请在服务端 .env 中配置。")
        if app.state.active_requests >= 4:
            raise HTTPException(429, "正在处理较多请求，请稍后重试。")
        app.state.active_requests += 1
        try:
            async with asyncio.timeout(settings.request_timeout_seconds):
                return await app.state.agent.run(body)
        except ProviderError as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})
        except ToolBudgetExceeded as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        except TimeoutError:
            return JSONResponse(status_code=504, content={"detail": "处理超时，请缩小问题后重试。"})
        finally:
            app.state.active_requests -= 1

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(PACKAGE_DIR / "static/index.html")

    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    return app


app = create_app()
