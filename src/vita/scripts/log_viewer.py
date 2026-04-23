import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vita.data_model.message import ToolCall
from vita.data_model.simulation import Results
from vita.environment.environment import get_cross_environment
from vita.registry import registry
from vita.utils.simulation_timeline import (
    build_timeline_from_simulation,
    build_user_simulator_profile,
    format_json_for_display,
    simulation_run_summary,
)
from vita.utils.utils import DATA_DIR


def _simulations_dir() -> Path:
    return DATA_DIR / "simulations"


def _resolve_simulation_file(file: str) -> Path:
    """仅允许读取 data/simulations 下的单个 .json 文件名，防止路径穿越。"""
    if not file or "/" in file or "\\" in file or file.startswith("."):
        raise HTTPException(status_code=400, detail="invalid simulation file name")
    if ".." in file:
        raise HTTPException(status_code=400, detail="invalid simulation file name")
    if not file.endswith(".json"):
        raise HTTPException(status_code=400, detail="simulation file must end with .json")
    root = _simulations_dir().resolve()
    path = (root / file).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside simulations dir") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"simulation file not found: {file}")
    return path


def _load_events(file_path: Path) -> list[dict]:
    events = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class ToolInvokeRequest(BaseModel):
    file: str
    sim_index: int
    tool_name: str
    arguments: dict = {}


def _resolve_task_from_simulation(results: Results, sim_index: int):
    if sim_index >= len(results.simulations):
        raise HTTPException(
            status_code=400,
            detail=f"sim_index out of range: {sim_index} (have {len(results.simulations)} simulations)",
        )
    sim = results.simulations[sim_index]
    task = next((t for t in results.tasks if t.id == sim.task_id), None)
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"task not found in results.tasks: {sim.task_id}"
        )
    return sim, task


def _build_environment_for_task(task, language: str = "chinese"):
    if "," in task.domain:
        return get_cross_environment(task.domain, task.environment, language)
    env_constructor = registry.get_env_constructor(task.domain)
    return env_constructor(task.environment, language)


def create_app(logs_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="VitaBench Log Viewer")
    base_logs_dir = Path(logs_dir) if logs_dir else DATA_DIR / "logs"
    static_dir = Path(__file__).parents[1] / "web"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        p = static_dir / "trajectory.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="trajectory.html not found")
        return p.read_text(encoding="utf-8")

    @app.get("/trajectory", response_class=HTMLResponse)
    def trajectory_page() -> str:
        p = static_dir / "trajectory.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="trajectory.html not found")
        return p.read_text(encoding="utf-8")

    @app.get("/events", response_class=HTMLResponse)
    def events_page() -> RedirectResponse:
        return RedirectResponse(url="/", status_code=307)

    @app.get("/toolbox", response_class=HTMLResponse)
    def toolbox_page() -> str:
        p = static_dir / "tools.html"
        if not p.exists():
            raise HTTPException(status_code=404, detail="tools.html not found")
        return p.read_text(encoding="utf-8")

    @app.get("/api/runs")
    def list_runs() -> dict:
        if not base_logs_dir.exists():
            return {"runs": []}
        runs = []
        for fp in sorted(base_logs_dir.glob("*.jsonl"), reverse=True):
            events = _load_events(fp)
            run_id = fp.stem
            total_events = len(events)
            simulations = sorted(
                {
                    f"{event.get('task_id')}#{event.get('trial')}"
                    for event in events
                    if event.get("task_id") is not None and event.get("trial") is not None
                }
            )
            runs.append(
                {
                    "run_id": run_id,
                    "file_name": fp.name,
                    "total_events": total_events,
                    "simulations": simulations,
                }
            )
        return {"runs": runs}

    @app.get("/api/events")
    def get_events(
        run_id: str = Query(...),
        task_id: Optional[str] = Query(None),
        trial: Optional[int] = Query(None),
        event_type: Optional[str] = Query(None),
        q: Optional[str] = Query(None),
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict:
        file_path = base_logs_dir / f"{run_id}.jsonl"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"run log not found: {run_id}")

        events = _load_events(file_path)
        filtered = []
        for event in events:
            if task_id is not None and event.get("task_id") != task_id:
                continue
            if trial is not None and event.get("trial") != trial:
                continue
            if event_type is not None and event.get("event_type") != event_type:
                continue
            if q is not None and q.strip():
                raw = json.dumps(event, ensure_ascii=False)
                if q.strip().lower() not in raw.lower():
                    continue
            filtered.append(event)

        return {
            "run_id": run_id,
            "count": len(filtered),
            "events": filtered[:limit],
            "truncated": len(filtered) > limit,
        }

    @app.get("/api/simulations")
    def list_simulation_files() -> dict:
        d = _simulations_dir()
        if not d.exists():
            return {"files": []}
        files = []
        for fp in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = fp.stat()
            files.append(
                {
                    "name": fp.name,
                    "mtime": st.st_mtime,
                    "size": st.st_size,
                }
            )
        return {"files": files}

    @app.get("/api/simulation-runs")
    def simulation_runs(file: str = Query(..., description="simulations 目录下的 json 文件名")) -> dict:
        path = _resolve_simulation_file(file)
        results = Results.model_validate_json(path.read_text(encoding="utf-8"))
        runs = []
        for i, sim in enumerate(results.simulations):
            runs.append({"index": i, **simulation_run_summary(sim)})
        return {
            "file": file,
            "task_count": len(results.tasks),
            "simulation_count": len(results.simulations),
            "runs": runs,
        }

    @app.get("/api/simulation-timeline")
    def simulation_timeline(
        file: str = Query(..., description="simulations 目录下的 json 文件名"),
        sim_index: int = Query(0, ge=0, description="results.simulations 中的下标"),
        include_raw: bool = Query(True, description="是否在每条 assistant/user 中包含 raw_data"),
    ) -> dict:
        path = _resolve_simulation_file(file)
        results = Results.model_validate_json(path.read_text(encoding="utf-8"))
        if sim_index >= len(results.simulations):
            raise HTTPException(
                status_code=400,
                detail=f"sim_index out of range: {sim_index} (have {len(results.simulations)} simulations)",
            )
        sim = results.simulations[sim_index]
        timeline = build_timeline_from_simulation(sim)
        if not include_raw:
            for row in timeline:
                if row.get("kind") in ("assistant", "user"):
                    row.pop("raw_data", None)
        return {
            "file": file,
            "sim_index": sim_index,
            "summary": simulation_run_summary(sim),
            "evaluation_result": sim.reward_info.model_dump(mode="json")
            if sim.reward_info is not None
            else None,
            "user_simulator_profile": build_user_simulator_profile(
                results, sim.task_id
            ),
            "timeline": timeline,
        }

    @app.get("/api/simulation-raw-json")
    def simulation_raw_json(
        file: str = Query(...),
        sim_index: int = Query(0, ge=0),
        section: str = Query(
            "messages",
            description="messages | full — full 为整份 Results JSON（可能很大）",
        ),
    ) -> dict:
        path = _resolve_simulation_file(file)
        raw_text = path.read_text(encoding="utf-8")
        if section == "full":
            return {
                "file": file,
                "format": "json",
                "text": format_json_for_display(json.loads(raw_text)),
            }
        results = Results.model_validate_json(raw_text)
        if sim_index >= len(results.simulations):
            raise HTTPException(status_code=400, detail="sim_index out of range")
        sim = results.simulations[sim_index]
        return {
            "file": file,
            "sim_index": sim_index,
            "format": "json",
            "text": format_json_for_display(
                [m.model_dump(mode="json") for m in sim.messages]
            ),
        }

    @app.get("/api/simulation-tools")
    def simulation_tools(
        file: str = Query(..., description="simulations 目录下的 json 文件名"),
        sim_index: int = Query(0, ge=0),
    ) -> dict:
        path = _resolve_simulation_file(file)
        results = Results.model_validate_json(path.read_text(encoding="utf-8"))
        sim, task = _resolve_task_from_simulation(results, sim_index)
        env = _build_environment_for_task(task)
        tools = []
        for t in env.get_tools():
            tools.append(
                {
                    "name": t.name,
                    "short_desc": t.short_desc,
                    "long_desc": t.long_desc,
                    "openai_schema": t.openai_schema,
                    "params_schema": t.params.model_json_schema(),
                    "returns_schema": t.returns.model_json_schema(),
                    "raises": t.raises,
                    "examples": t.examples,
                }
            )
        tools.sort(key=lambda x: x["name"])
        return {
            "file": file,
            "sim_index": sim_index,
            "task_id": sim.task_id,
            "domain": task.domain,
            "note": "工具调用基于该任务的初始 environment 状态，不会自动回放到某一轮对话后的中间状态。",
            "tools": tools,
        }

    @app.post("/api/simulation-tool-invoke")
    def simulation_tool_invoke(req: ToolInvokeRequest) -> dict:
        path = _resolve_simulation_file(req.file)
        results = Results.model_validate_json(path.read_text(encoding="utf-8"))
        sim, task = _resolve_task_from_simulation(results, req.sim_index)
        env = _build_environment_for_task(task)
        tool_msg = env.get_response(
            ToolCall(
                id=f"debug_{uuid.uuid4().hex[:12]}",
                name=req.tool_name,
                arguments=req.arguments or {},
                requestor="assistant",
            )
        )
        return {
            "file": req.file,
            "sim_index": req.sim_index,
            "task_id": sim.task_id,
            "domain": task.domain,
            "tool_message": tool_msg.model_dump(mode="json"),
        }

    return app
