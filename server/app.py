"""
Server entry point for OpenEnv deployment.
Exposes the FastAPI app and a main() function for the project.scripts entry point.
"""

import uvicorn
from env.server import app


def main():
    """Run the MedRecordAudit server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
