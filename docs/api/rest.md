# REST API

702 routes across 75 tag groups. The schema below is generated from the running FastAPI application by
`scripts/build_docs.py` on every site build, so it cannot drift from the code.

!!! warning "Response schemas are thin"
    Only a handful of routes declare a `response_model`. The request side, path and query parameters, and
    status codes are accurate; most response bodies are not described. The generated Python SDK
    (`sdk/generated_client.py`) is correspondingly untyped. This is a known gap, not an oversight.

!!! info "Authentication"
    Every route sits behind a bearer token with one of three roles - `annotator`, `reviewer`, `admin` -
    enforced fail-closed. Routers declare a floor with `require_role(...)`; a route with no explicit floor
    still requires a valid token. Mint one on the box with
    `python -m scripts.mint_token --name you --role admin --create`.

[Download openapi.json](openapi.json){ .md-button }

<div id="redoc"></div>
<script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
<script>
  Redoc.init('openapi.json', {
    scrollYOffset: 60,
    hideDownloadButton: false,
    theme: { typography: { fontSize: '14px' } }
  }, document.getElementById('redoc'));
</script>
