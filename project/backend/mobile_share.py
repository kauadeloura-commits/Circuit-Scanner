"""Expose only token-authorized share views and simulator assets on the LAN."""
from fastapi import FastAPI
from starlette.routing import Mount

def mobile_app(main_app):
    app=FastAPI(title='Circuit Scanner · celular',docs_url=None,redoc_url=None,openapi_url=None)
    for route in main_app.routes:
        path=getattr(route,'path','')
        if path.startswith('/share/') and getattr(route,'methods',set())=={'GET'}:
            app.router.routes.append(route)
        elif path=='/visual-editor.js' and getattr(route,'methods',set())=={'GET'}:
            # The shared simulation page loads this read-only UI asset too.
            # Project editing routes remain restricted to the local server.
            app.router.routes.append(route)
        elif isinstance(route,Mount) and path=='/simulators/circuitjs':
            app.router.routes.append(route)
    @app.get('/health')
    def health():return {'ready':True,'mode':'shared_circuit_only'}
    return app
