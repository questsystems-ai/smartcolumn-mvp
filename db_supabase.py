# db_supabase.py
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # pick up .env in dev

# Prefer service role if present (server-side only), else fall back to anon
def get_supabase_keys():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL and/or SUPABASE_*_KEY in env")
    return url, key

def get_sb():
    from supabase import create_client
    url, key = get_supabase_keys()
    return create_client(url, key)
