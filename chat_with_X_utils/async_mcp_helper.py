"""A helper class to manage async MCP sessions in Jupyter Notebooks across cells"""

import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools


class MCPManager:
    def __init__(self, client):
        self.client = client
        self._sessions = {}
        self._tasks = {}
        self._tools = {}

    async def _session_runner(self, name, ready_event):
        async with self.client.session(name) as session:
            self._sessions[name] = session
            self._tools[name] = await load_mcp_tools(session)
            ready_event.set()
            await asyncio.Future()  # keep alive

    async def start_session(self, name: str):
        if name in self._tasks:
            raise RuntimeError(f"Session {name!r} already running")
        ready_event = asyncio.Event()
        task = asyncio.create_task(self._session_runner(name, ready_event))
        self._tasks[name] = task
        await ready_event.wait()
        return self._sessions[name], self._tools[name]

    async def stop_session(self, name: str):
        task = self._tasks.pop(name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._sessions.pop(name, None)
        self._tools.pop(name, None)

    def get_session(self, name: str):
        return self._sessions.get(name)

    def get_tools(self, name: str):
        return self._tools.get(name)

    async def stop_all(self):
        for name in list(self._tasks.keys()):
            await self.stop_session(name)
