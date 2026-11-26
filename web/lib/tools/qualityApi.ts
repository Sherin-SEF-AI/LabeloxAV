// Client for the quality annotation tools (server: services/api/routers/tools_quality.py, paths /api/tools/quality/...).
// Build calls on apiGet/apiPost/apiPut/apiDelete; start background work through a route that returns a ToolJob
// and follow it with useToolJob from ./toolJobs.

export {};
