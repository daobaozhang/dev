# web_app.py
import os
import requests
import json
import logging
import time
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sqlite3
from dotenv import load_dotenv

# 配置日志
os.makedirs("logs", exist_ok=True)  # 创建日志目录

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FreeLLMScanner")

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- API Configuration (Environment-based) ---
API_HTTP_REFERER = os.getenv("API_HTTP_REFERER", "http://localhost:8000")
API_TITLE = os.getenv("API_TITLE", "Free LLM Scanner")
API_USER_AGENT = os.getenv("API_USER_AGENT", "FreeLLMScanner/2.0")

# --- Database Setup ---
DB_PATH = "data/visitor_stats.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS visits
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ip TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- Middleware for Visitor Tracking ---
@app.middleware("http")
async def track_visits(request: Request, call_next):
    response = await call_next(request)
    
    # Only track successful requests to the root page to avoid spamming DB with static files/API calls
    if request.url.path == "/" and response.status_code == 200:
        client_ip = request.client.host
        # Handle proxy headers if behind Nginx/Docker
        if "x-forwarded-for" in request.headers:
            client_ip = request.headers["x-forwarded-for"].split(",")[0]
            
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO visits (ip) VALUES (?)", (client_ip,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error tracking visit: {e}")
            
    return response

# --- Rate Limiting Middleware ---
rate_limit_store = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # 只对API端点应用速率限制
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0]
    
    key = f"{client_ip}:{request.url.path}"
    now = time.time()
    window = 60  # 60秒窗口
    
    if key not in rate_limit_store:
        rate_limit_store[key] = []
    
    # 清理过期记录
    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < window]
    
    # 检查速率限制 (每60秒最多30次)
    if len(rate_limit_store[key]) >= 30:
        return {"status": "error", "message": "请求过于频繁，请稍后再试", "code": "RATE_LIMIT_EXCEEDED"}
    
    rate_limit_store[key].append(now)
    
    return await call_next(request)

# --- Default Providers Configuration ---
# 所有模型都只推荐免费的！
PROVIDERS = {
    "openrouter": { 
        "base_url": "https://openrouter.ai/api/v1",
        "name": "OpenRouter (聚合平台)",
        "description": "全球最大的 LLM 聚合平台。集成数百种免费模型。",
        "website": "https://openrouter.ai",
        "requires_key": True,
        "recommended_models": [
            # 2026年2月 OpenRouter 免费模型 (全部免费)
            {"id": "openrouter/free", "name": "Auto Select (自动选择)", "is_free": True, "tags": ["💬 对话"]},
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "qwen/qwen-2.5-72b-instruct:free", "name": "Qwen 2.5 72B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1", "is_free": True, "tags": ["🧠 深度思考"]}
        ]
    },
    "deepseek": { 
        "base_url": "https://api.deepseek.com",
        "name": "DeepSeek (深度求索)",
        "description": "国产最强开源模型。注册送 500 万 token。",
        "website": "https://www.deepseek.com",
        "requires_key": True,
        # DeepSeek 注册送免费额度
         "recommended_models": [
            {"id": "deepseek-chat", "name": "DeepSeek-V3 (有免费额度)", "is_free": True, "tags": ["💬 对话"]},
            {"id": "deepseek-reasoner", "name": "DeepSeek-R1 (有免费额度)", "is_free": True, "tags": ["🧠 深度思考"]}
        ]
    },
    "siliconflow": { 
        "base_url": "https://api.siliconflow.cn/v1",
        "name": "SiliconFlow (硅基流动)",
        "description": "高性能推理平台。部分模型永久免费。",
        "website": "https://siliconflow.cn",
        "requires_key": True,
        "recommended_models": [
            # 2026年2月 免费的模型
            {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "THUDM/glm-4-flash", "name": "GLM-4-Flash", "is_free": True, "tags": ["💬 对话"]},
            {"id": "THUDM/glm-4-9b-chat", "name": "GLM-4-9B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek-V3", "is_free": True, "tags": ["💬 对话"]}
        ]
    },
    "zhipu": { 
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "name": "Zhipu AI (智谱)",
        "description": "清华系大模型。GLM-4-Flash 永久免费。",
        "website": "https://open.bigmodel.cn",
        "requires_key": True,
        "recommended_models": [
            {"id": "glm-4-flash", "name": "GLM-4-Flash (永久免费)", "is_free": True, "tags": ["💬 对话"]},
            {"id": "glm-4-flash-0711", "name": "GLM-4-Flash-0711", "is_free": True, "tags": ["💬 对话"]},
            {"id": "glm-3-turbo", "name": "GLM-3-Turbo", "is_free": True, "tags": ["💬 对话"]}
        ]
    },
    "groq": { 
        "base_url": "https://api.groq.com/openai/v1",
        "name": "Groq (极速推理)",
        "description": "全球最快推理速度。大量免费额度。",
        "website": "https://groq.com",
        "requires_key": True,
        "recommended_models": [
            {"id": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "llama3-70b-8192", "name": "Llama 3 70B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "is_free": True, "tags": ["💬 对话"]}
        ]
    },
    "gemini": { 
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "name": "Google Gemini",
        "description": "Google 最强模型。免费层有额度。",
        "website": "https://ai.google.dev",
        "requires_key": True,
        "recommended_models": [
             # 使用 gemini-2.0-flash (有免费额度)
             {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash", "is_free": True, "tags": ["💬 对话", "🖼️ 视觉"]},
             {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B", "is_free": True, "tags": ["💬 对话", "🖼️ 视觉"]},
             {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "is_free": True, "tags": ["💬 对话", "🖼️ 视觉"]}
        ]
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "name": "NVIDIA NIM",
        "description": "NVIDIA 官方推理。注册送免费额度。",
        "website": "https://build.nvidia.com",
        "requires_key": True,
        "recommended_models": [
             {"id": "nvidia/llama-3.1-nemotron-70b-instruct", "name": "Nemotron 70B", "is_free": True, "tags": ["💬 对话"]},
             {"id": "nvidia/llama-3.1-8b-instruct", "name": "Llama 3.1 8B", "is_free": True, "tags": ["💬 对话"]},
             {"id": "microsoft/phi-3-mini-128k-instruct", "name": "Phi-3 Mini", "is_free": True, "tags": ["💬 对话"]}
        ]
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "name": "SambaNova Cloud",
        "description": "高性能平台。目前免费使用。",
        "website": "https://sambanova.ai",
        "requires_key": True,
        "recommended_models": [
             {"id": "Meta-Llama-3.1-405B-Instruct", "name": "Llama 3.1 405B", "is_free": True, "tags": ["💬 对话"]},
             {"id": "Meta-Llama-3.1-70B-Instruct", "name": "Llama 3.1 70B", "is_free": True, "tags": ["💬 对话"]},
             {"id": "Meta-Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B", "is_free": True, "tags": ["💬 对话"]}
        ]
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/models",
        "name": "Hugging Face",
        "description": "开源社区。免费层适合测试。",
        "website": "https://huggingface.co",
        "requires_key": True,
        "recommended_models": [
            {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "name": "Llama 3 8B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B", "is_free": True, "tags": ["💬 对话"]},
            {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini", "is_free": True, "tags": ["💬 对话"]}
        ]
    }
}


class AiQuery(BaseModel):
    query: str
    apiKey: str
    providerId: str = "openrouter" 
    modelId: str = "openrouter/free"  # 默认使用自动选择免费模型
    tavilyKey: Optional[str] = None

# --- Helpers ---
def perform_tavily_search(query: str, api_key: str):
    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5}
        res = requests.post(url, json=payload, timeout=15)
        if res.status_code == 200:
            results = res.json().get("results", [])
            context = "\n".join([f"- {r['title']}: {r['content']} (URL: {r['url']})" for r in results])
            return context
        return f"(Search Error: HTTP {res.status_code})"
    except Exception as e: return f"(Search Error: {str(e)})"

# --- Fetch Helpers (Code Deduplication) ---
DEFAULT_TIMEOUT = 10

def _fetch_models_from_api(base_url, headers=None, timeout=DEFAULT_TIMEOUT):
    """
    统一的API模型获取函数
    返回: {"status": "success"|"error", "data": ..., "message": ...}
    """
    try:
        response = requests.get(base_url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        return {"status": "error", "message": f"HTTP {response.status_code}", "details": response.text[:200]}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "请求超时", "details": f"超过 {timeout}s"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "网络连接失败", "details": "请检查网络设置"}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"HTTP错误", "details": str(e)}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": "响应格式错误", "details": "JSON解析失败"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": "请求错误", "details": str(e)}
    except Exception as e:
        return {"status": "error", "message": "未知错误", "details": str(e)}

def _build_fallback_models(recommended_models, context="?", free_only=True):
    """构建静态回退模型列表
    
    Args:
        recommended_models: 模型列表
        context: 上下文长度描述
        free_only: 是否只返回免费模型 (默认True)
    """
    filtered = [
        m for m in recommended_models
        if not free_only or m.get("is_free", False)  # 只返回免费模型
    ]
    
    return [
        {
            "id": m["id"],
            "name": m["name"],
            "context": context,
            "is_free": m.get("is_free", False)
        }
        for m in filtered
    ]


def get_openrouter_models(api_key=None):
    # If a key is provided, we can verify it using /auth/key
    source_type = "public_api"
    if api_key:
        try:
            auth_url = f"{PROVIDERS['openrouter']['base_url']}/auth/key"
            # OpenRouter often requires these headers
            h = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": API_HTTP_REFERER,
                "X-Title": API_TITLE,
                "User-Agent": API_USER_AGENT
            }
            auth_res = requests.get(auth_url, headers=h, timeout=10)
            if auth_res.status_code == 200:
                source_type = "live_key" # Valid key
        except requests.exceptions.Timeout:
            logger.warning(f"OpenRouter 密钥验证超时")
        except requests.exceptions.ConnectionError:
            logger.warning(f"OpenRouter 密钥验证网络连接失败")
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenRouter 密钥验证请求错误: {e}")
        except Exception as e:
            logger.error(f"OpenRouter 密钥验证未知错误: {e}")
    
    # Fetch public models (OpenRouter public list is better than authenticated list which might be empty if user has no access?)
    # OpenRouter /models is public。
    url = f"{PROVIDERS['openrouter']['base_url']}/models"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            all_models = data.get("data", [])
            free_models = []
            for m in all_models:
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", -1))
                completion_price = float(pricing.get("completion", -1))
                
                # 只筛选完全免费的模型 (prompt和completion价格都为0)
                if prompt_price == 0 and completion_price == 0:
                    free_models.append({
                        "id": m["id"],
                        "name": m["name"],
                        "context": str(m.get("context_length", 0)),
                        "pricing": pricing,
                        "is_free": True
                    })
            
            # 按上下文长度降序排序，优先展示能力强的免费模型
            free_models.sort(key=lambda x: int(x.get("context", 0)), reverse=True)
            
            logger.info(f"OpenRouter: Found {len(free_models)} free models")
            return {"status": "success", "models": free_models, "source": source_type}
    except requests.exceptions.Timeout:
        logger.warning(f"OpenRouter API 请求超时")
        return {"status": "error", "message": "API请求超时，请稍后重试"}
    except requests.exceptions.ConnectionError:
        logger.warning(f"OpenRouter API 网络连接失败")
        return {"status": "error", "message": "网络连接失败，请检查网络设置"}
    except requests.exceptions.HTTPError as e:
        logger.warning(f"OpenRouter API HTTP错误: {e}")
        return {"status": "error", "message": f"HTTP错误: {e.response.status_code}"}
    except json.JSONDecodeError as e:
        logger.warning(f"OpenRouter API 响应解析失败: {e}")
        return {"status": "error", "message": "响应格式错误，请稍后重试"}
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenRouter API 请求错误: {e}")
        return {"status": "error", "message": f"请求错误: {str(e)}"}
    except Exception as e:
        logger.error(f"OpenRouter API 未知错误: {e}")
        return {"status": "error", "message": "内部服务器错误"}
    
    # Fallback: 只返回推荐的免费模型
    free_recommended = _build_fallback_models(
        PROVIDERS['openrouter']['recommended_models'], 
        context="?",
        free_only=True
    )
    logger.info(f"OpenRouter: Using fallback with {len(free_recommended)} free recommended models")
    return {"status": "success", "models": free_recommended, "source": "static_fallback", "message": "Showing Free Recommended"}

def get_provider_models(provider_id, api_key=None):
    """Generic fetcher for standard OpenAI-compatible providers."""
    config = PROVIDERS.get(provider_id)
    if not config: return {"status": "error", "message": "Unknown Provider"}
    
    # Static Fallback if no key - 只返回免费模型
    if not api_key:
        return {
            "status": "success", 
            "models": _build_fallback_models(config['recommended_models'], free_only=True), 
            "source": "static_fallback", 
            "message": "No API Key - Showing Free Models"
        }

    # Live Fetch
    url = f"{config['base_url']}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    if provider_id == "siliconflow":
         # Siliconflow /models is public? No, usually needs key.
         pass
    
    # 使用统一辅助函数
    result = _fetch_models_from_api(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    
    if result["status"] == "success":
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}
        # Standardize OpenAI format: data: [{id: ...}, ...]
        models = []
        for m in data.get("data", []):
            # 尝试过滤免费模型
            # SiliconFlow: 检查是否有pricing信息
            is_free = False
            if provider_id == "siliconflow":
                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", -1))
                completion_price = float(pricing.get("completion", -1))
                is_free = (prompt_price == 0 and completion_price == 0)
            
            # 过滤掉已知的付费模型模式
            lower_id = m["id"].lower()
            paid_patterns = ["pro/", "kwaipilot/", "-pro", "premium/", "advanced/"]
            is_paid = any(pattern in lower_id for pattern in paid_patterns)
            
            if not is_paid or is_free:
                model_info = {
                    "id": m["id"],
                    "name": m["id"].split('/')[-1], # Clean name
                    "context": "?",
                    "is_free": is_free or not is_paid
                }
                models.append(model_info)
        
        logger.info(f"{provider_id}: Fetched {len(models)} models from API (includes both free and paid)")
        return {"status": "success", "models": models, "source": "live_key"}
    else:
        logger.warning(f"Fetch Models Error {provider_id}: {result['message']} - {result.get('details', '')}")
    
    # Fallback to recommended - 只返回免费模型
    free_models = _build_fallback_models(config['recommended_models'], free_only=True)
    if len(free_models) == 0:
        # 如果没有免费模型，返回空列表并提示
        return {
            "status": "success", 
            "models": [], 
            "source": "static_fallback_on_error", 
            "message": f"No free models available for {config['name']}"
        }
    
    return {
        "status": "success", 
        "models": free_models, 
        "source": "static_fallback_on_error", 
        "message": f"API Error - Showing {len(free_models)} Free Models"
    }


class LatencyRequest(BaseModel):
    provider_id: str
    model_id: str
    api_key: Optional[str] = None

import time

@app.post("/api/test_model")
async def test_model_latency(req: LatencyRequest):
    """测速功能 - 仅OpenRouter可用，其他厂商返回不支持"""
    provider = PROVIDERS.get(req.provider_id)
    if not provider:
        return {"status": "error", "message": "Unknown Provider"}

    # 只有 OpenRouter 支持测速功能
    if req.provider_id != "openrouter":
        return {"status": "error", "message": "测速仅支持 OpenRouter"}

    api_key = req.api_key
    if not api_key and provider.get('requires_key', True):
        return {"status": "error", "message": "Missing Key"}

    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": API_USER_AGENT,
        "HTTP-Referer": API_HTTP_REFERER,
        "X-Title": API_TITLE
    }

    # Minimal payload for speed test
    payload = {
        "model": req.model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    }

    start_time = time.time()
    try:
        logger.info(f"test_model_latency: Testing {req.provider_id} - {req.model_id}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(f"test_model_latency: {req.provider_id} responded {response.status_code} in {latency_ms}ms")

        if response.status_code == 200:
            return {"status": "success", "latency": latency_ms}
        elif response.status_code in [401, 403]:
            error_detail = ""
            try:
                err_data = response.json()
                error_detail = err_data.get("error", {}).get("message", response.text[:100])
            except:
                error_detail = response.text[:100]
            logger.warning(f"test_model_latency: Auth error for {req.provider_id}: {error_detail}")
            return {"status": "error", "message": "Auth/Paid", "latency": latency_ms}
        elif response.status_code == 429:
            logger.warning(f"test_model_latency: Rate limited for {req.provider_id}")
            return {"status": "error", "message": "RateLimited", "latency": latency_ms}
        else:
            error_detail = ""
            try:
                err_data = response.json()
                error_detail = str(err_data)[:200]
            except:
                error_detail = response.text[:200]
            logger.error(f"test_model_latency: Error {response.status_code} for {req.provider_id}: {error_detail}")
            return {"status": "error", "message": f"{response.status_code}: {error_detail[:50]}", "latency": latency_ms}

    except requests.exceptions.ConnectTimeout:
        logger.error(f"test_model_latency: Timeout for {req.provider_id}")
        return {"status": "error", "message": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"test_model_latency: Connection error for {req.provider_id}: {e}")
        return {"status": "error", "message": "NetErr"}
    except Exception as e:
        logger.exception(f"test_model_latency: Exception for {req.provider_id}: {e}")
        return {"status": "error", "message": f"Error: {str(e)}"}

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 添加 favicon 路由 - 返回实际的图标文件
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    favicon_path = os.path.join(base_dir, "static", "favicon.ico")
    
    if os.path.exists(favicon_path):
        from fastapi.responses import FileResponse
        return FileResponse(favicon_path, media_type="image/x-icon")
    
    # 如果文件不存在，返回一个简单的空响应
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/api/stats")
async def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total visits
        c.execute("SELECT COUNT(*) FROM visits")
        total_visits = c.fetchone()[0]
        
        # Unique IPs
        c.execute("SELECT COUNT(DISTINCT ip) FROM visits")
        unique_visitors = c.fetchone()[0]
        
        conn.close()
        return {"total_visits": total_visits, "unique_visitors": unique_visitors}
    except Exception as e:
        return {"total_visits": 0, "unique_visitors": 0, "error": str(e)}

@app.get("/api/providers")
async def get_providers():
    return PROVIDERS


# --- Request Models ---
class AiQuery(BaseModel):
    query: str
    apiKey: str
    providerId: str
    modelId: str
    tavilyKey: Optional[str] = None


@app.post("/api/ask_ai")
async def ask_ai(query: AiQuery):
    if not query.apiKey: 
        logger.warning("ask_ai: Missing API key")
        raise HTTPException(status_code=400, detail="API Key required")
    
    # 验证 modelId
    if not query.modelId:
        logger.warning("ask_ai: Missing modelId")
        raise HTTPException(status_code=400, detail="Model ID is required")
    
    logger.info(f"ask_ai: Processing request - provider={query.providerId}, model={query.modelId}")
    
    search_context = ""
    if query.tavilyKey:
        logger.info("ask_ai: Using Tavily search")
        search_context = perform_tavily_search(query.query, query.tavilyKey)
        search_context = f"\n\n[Tavily Search Results]:\n{search_context}\n"
    else:
        # 没有 Tavily Key 时，使用内置的免费模型信息
        logger.info("ask_ai: No Tavily key, using built-in provider info")
        search_context = "\n\n[BUILT-IN FREE PROVIDERS INFO]:\n"
        for pid, pdata in PROVIDERS.items():
            search_context += f"\n- {pdata['name']}: {pdata['description']}\n"
            if pdata.get('recommended_models'):
                models = [m['name'] for m in pdata['recommended_models'] if m.get('is_free')]
                search_context += f"  Free models: {', '.join(models)}\n"

    provider_config = PROVIDERS.get(query.providerId)
    if not provider_config:
        logger.error(f"ask_ai: Unknown provider {query.providerId}")
        raise HTTPException(status_code=400, detail=f"Unknown provider: {query.providerId}")
    
    url = f"{provider_config['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {query.apiKey}", "Content-Type": "application/json"}
    
    # 为所有provider添加必要的headers
    if query.providerId == "openrouter":
        headers["HTTP-Referer"] = API_HTTP_REFERER
        headers["X-Title"] = API_TITLE
    headers["User-Agent"] = API_USER_AGENT

    # STRICT JSON SYSTEM PROMPT -> ENHANCED HYBRID PROMPT
    system_prompt = (
        "You are the AI Assistant for 'Free LLM Scanner', a platform that helps users find free LLM APIs. "
        "Your capabilities: \n"
        "1. Answer general questions about LLMs, APIs, and this platform. \n"
        "2. Guide users on how to use this platform (Key is stored locally in localStorage, supports OpenRouter/DeepSeek/etc). \n"
        "3. Find resources based on user search. \n\n"
        "CRITICAL INSTRUCTION: \n"
        "- If the user asks a question, answer it helpfully in natural language (Markdown supported). \n"
        "- If (and ONLY if) the user is looking for specific LLM resources/models, provide a list. \n"
        "- **STRICTLY FREE MODELS ONLY**: The user explicitly requested ONLY free models. Do NOT recommend any model that has a cost per token. \n"
        "- **ALWAYS provide resources in JSON format at the END of your response**. \n"
        "- The JSON must be in a code block with ```json markers. \n"
        "- Include ALL free providers from the built-in list. \n"
        "Format for resources (MUST include this JSON block): \n"
        "```json\n"
        '[{"id": "openrouter", "name": "OpenRouter (聚合平台)", "base_url": "https://openrouter.ai/api/v1", "description": "全球最大的LLM聚合平台，集成数百种免费模型", "recommended_models": [{"id": "openrouter/free", "name": "Auto Select"}, {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"}, {"id": "deepseek-ai/DeepSeek-R1", "name": "DeepSeek R1"}]}, {"id": "deepseek", "name": "DeepSeek (深度求索)", "base_url": "https://api.deepseek.com", "description": "国产最强开源模型，注册送500万token", "recommended_models": [{"id": "deepseek-chat", "name": "DeepSeek-V3"}, {"id": "deepseek-reasoner", "name": "DeepSeek-R1"}]}, {"id": "siliconflow", "name": "SiliconFlow (硅基流动)", "base_url": "https://api.siliconflow.cn/v1", "description": "高性能推理平台，部分模型永久免费", "recommended_models": [{"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B"}, {"id": "THUDM/glm-4-flash", "name": "GLM-4-Flash"}]}, {"id": "zhipu", "name": "Zhipu AI (智谱)", "base_url": "https://open.bigmodel.cn/api/paas/v4", "description": "清华系大模型，GLM-4-Flash永久免费", "recommended_models": [{"id": "glm-4-flash", "name": "GLM-4-Flash"}]}, {"id": "groq", "name": "Groq (极速推理)", "base_url": "https://api.groq.com/openai/v1", "description": "全球最快推理速度，大量免费额度", "recommended_models": [{"id": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B"}, {"id": "gemma2-9b-it", "name": "Gemma 2 9B"}]}, {"id": "gemini", "name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "description": "Google最强模型，免费层有额度", "recommended_models": [{"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"}]}, {"id": "nvidia", "name": "NVIDIA NIM", "base_url": "https://integrate.api.nvidia.com/v1", "description": "NVIDIA官方推理，注册送免费额度", "recommended_models": [{"id": "nvidia/llama-3.1-8b-instruct", "name": "Llama 3.1 8B"}]}, {"id": "sambanova", "name": "SambaNova Cloud", "base_url": "https://api.sambanova.ai/v1", "description": "高性能平台，目前免费使用", "recommended_models": [{"id": "Meta-Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B"}]}, {"id": "huggingface", "name": "Hugging Face", "base_url": "https://api-inference.huggingface.co/models", "description": "开源社区，免费层适合测试", "recommended_models": [{"id": "meta-llama/Meta-Llama-3-8B-Instruct", "name": "Llama 3 8B"}]}]\n'
        "```"
    )

    payload = {
        "model": query.modelId, 
        "messages": [{"role": "system", "content": system_prompt + search_context}, {"role": "user", "content": query.query}],
        "temperature": 0.3, 
        "max_tokens": 2000
    }
    
    logger.info(f"ask_ai: Calling {url}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        logger.info(f"ask_ai: Response status={response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"ask_ai: Response data keys={list(data.keys())}")
            
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                logger.info(f"ask_ai: Success - content length={len(content)}")
                return {"status": "success", "answer": content}
            
            # 如果没有 choices，返回错误
            logger.warning(f"ask_ai: Empty choices in response: {str(data)[:300]}")
            return {"status": "error", "message": "Model returned empty response. Please try a different model."}
        
        # 处理 HTTP 错误
        logger.error(f"ask_ai: HTTP Error {response.status_code}: {response.text[:200]}")
        return {"status": "error", "message": f"API Error {response.status_code}: {response.text[:100]}"}
        
    except requests.exceptions.Timeout:
        logger.error("ask_ai: Request timeout")
        return {"status": "error", "message": "Request timeout (60s)"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"ask_ai: Connection error: {e}")
        return {"status": "error", "message": f"Connection error: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"ask_ai: Invalid JSON response: {e}")
        return {"status": "error", "message": f"Invalid response from API"}
    except Exception as e:
        logger.exception(f"ask_ai: Unexpected error: {e}")
        return {"status": "error", "message": f"Server error: {str(e)}"}

@app.get("/api/verify/tavily")
def verify_tavily(request: Request):
    api_key = request.headers.get("x-api-key")
    if not api_key: return {"status": "error"}
    try:
        res = requests.post("https://api.tavily.com/search", json={"api_key": api_key, "query": "test", "max_results": 1}, timeout=5)
        if res.status_code == 200: return {"status": "success"}
        return {"status": "error", "code": res.status_code}
    except: return {"status": "error"}

@app.get("/api/verify/openrouter")
def verify_openrouter(request: Request):
    api_key = request.headers.get("x-api-key")
    if not api_key: return {"status": "error", "message": "No key"}
    
    try:
        # Relaxed Verification: Try Chat Completion directly (Auth endpoint can be flaky)
        # Using a very cheap/free model for verification or just listing models/auth
        # Standard Auth Check first
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Free LLM Scanner"
        }
        
        # 1. Try key info endpoint (fastest)
        res = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10)
        if res.status_code == 200:
            return {"status": "success", "data": res.json()}
            
        # 2. If 401/403, it's definitely invalid
        if res.status_code in [401, 403]:
             return {"status": "error", "message": "Invalid Key"}
             
        # 3. Network error or other? Assume success if we traveled this far but got 5xx? 
        # No, better to be strict. But user said "API is correct".
        # Let's trust 200 OK.
        return {"status": "error", "code": res.status_code, "body": res.text}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models/openrouter")
def api_openrouter(): return get_openrouter_models()

@app.get("/api/models/{provider_id}")
def api_generic_provider(provider_id: str, request: Request):
    return get_provider_models(provider_id, request.headers.get("x-api-key"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
