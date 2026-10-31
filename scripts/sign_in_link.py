"""Issue a sign-in link for a person, from the server itself: how the first administrator gets in, and how a
locked-out one gets back.

The installer used to print a raw API token and say "open /login and paste it", but the sign-in page has
no box for a token: it signs people in with a password. The seeded `admin` account has none, so a fresh
install's administrator had no visible way in.

A link works the way Settings > Team works for everyone else: it opens "choose a password" at /login, works
once, and expires (settings.auth.invite_ttl_hours). This runs on the box, so possession of the database and
signing key is the authority, the same as scripts/mint_token.py; that is why it is a script, not an endpoint.

    python -m scripts.sign_in_link --name admin
    python -m scripts.sign_in_link --name admin --url http://labelox.lab:3000
    python -m scripts.sign_in_link --name alice --role reviewer --create

Prints the link (or just the token with --token-only, for scripts that build the URL themselves).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import timedelta
from urllib.parse import quote

from sqlalchemy import select

from core.config import get_settings
from core.logging import setup_logging
from db.models import User
from db.session import get_sessionmaker
from services.api import identity as ident


async def _run(args: argparse.Namespace) -> int:
    hours = get_settings().auth.invite_ttl_hours
    async with get_sessionmaker()() as db:
        user = (await db.execute(select(User).where(User.name == args.name))).scalar_one_or_none()
        if user is None:
            if not args.create:
                print(f"no user named {args.name!r}. Pass --create to make one.", file=sys.stderr)
                return 1
            user = User(name=args.name, role=args.role)
            db.add(user)
            await db.commit()
            await db.refresh(user)
        token = await ident.create_reset(db, user, ttl=timedelta(hours=hours))
    if args.token_only:
        print(token)
    else:
        print(f"{args.url.rstrip('/')}/login?invite={quote(token)}")
        print(f"for {user.name} ({user.role}); works once, expires in {hours} hours", file=sys.stderr)
    return 0


def main() -> None:
    # Logs to stderr: stdout is the link, and the installer captures it.
    setup_logging(get_settings().log_level, stream=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--url", default="http://localhost:3000", help="the web app's address, for the printed link")
    p.add_argument("--create", action="store_true", help="create the user if missing")
    p.add_argument("--role", default="annotator", choices=["annotator", "reviewer", "admin"])
    p.add_argument("--token-only", action="store_true", help="print only the token")
    raise SystemExit(asyncio.run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
