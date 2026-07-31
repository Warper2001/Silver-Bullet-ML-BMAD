import asyncio
import os
import sys
import httpx
from pathlib import Path

base_dir = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(base_dir))
from src.research.projectx_auth import ProjectXAuth

async def explore_px():
    px_auth = ProjectXAuth.from_file(str(base_dir / '.projectx_api_key'))
    token = await px_auth.authenticate()
    
    px_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        # Try getting the user profile or accounts list
        resp = await client.get("https://api.topstepx.com/api/Account", headers=px_headers)
        print("GET /api/Account ->", resp.status_code, resp.text[:200])
        
        resp = await client.get("https://api.topstepx.com/api/Accounts", headers=px_headers)
        print("GET /api/Accounts ->", resp.status_code, resp.text[:200])

asyncio.run(explore_px())
