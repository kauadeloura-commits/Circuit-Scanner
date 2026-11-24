"""Optional in-memory progress; never participates in vision decisions."""
from contextvars import ContextVar

REPORTER=ContextVar('scanner_processing_reporter',default=None)

def report(label,completed=None,total=None):
    callback=REPORTER.get()
    if callback is not None:callback(label,completed,total)
