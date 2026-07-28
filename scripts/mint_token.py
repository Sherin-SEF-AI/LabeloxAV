"""Mint an API token for a user, from the server itself.

The recovery path for the situation the deployment otherwise has no answer to: every credential is issued
through the API, issuing one requires an admin token, and the bootstrap that let the first user be created
without one closes as soon as that user exists. Lose the last admin token and the only remaining options
were editing the database by hand or wiping the install.

This runs on the box, so possession of the signing key is the authority, which is the same authority the API
itself has. That is why it is a script and not an endpoint: an endpoint that minted tokens without
authentication would be the hole this is meant to avoid.

    python -m scripts.mint_token --name admin
    python -m scripts.mint_token --name alice --role reviewer --create
    python -m scripts.mint_token --name admin --revoke-existing
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from core.config import get_settings
from db.models import User
from db.session import get_sessionmaker
from services.api.auth_token import mint_token


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    async with get_sessionmaker()() as db:
        user = (await db.execute(select(User).where(User.name == args.name))).scalar_one_or_none()

        if user is None:
            if not args.create:
                print(f"no user named {args.name!r}. Pass --create to make one, or list users with:\n"
                      f"  python -m scripts.mint_token --list", file=sys.stderr)
                return 1
            user = User(name=args.name, role=args.role)
            db.add(user)
            await db.commit()
            await db.refresh(user)
            print(f"created user {user.name!r} with role {user.role!r}")

        if args.revoke_existing:
            # Bumping the version invalidates every token already issued to this user. The one printed below
            # is minted after the bump, so it is the only one that works: this is the "a token leaked" path.
            user.token_version = int(user.token_version or 1) + 1
            await db.commit()
            await db.refresh(user)
            print(f"revoked all previous tokens for {user.name!r} (token_version now {user.token_version})")

        token = mint_token(user.user_id, settings.auth.signing_key,
                           token_version=user.token_version,
                           ttl_seconds=settings.auth.token_ttl_seconds)
        hours = settings.auth.token_ttl_seconds / 3600
        print(f"\nuser  {user.name}  ({user.role})")
        print(f"valid {hours:.0f}h from now")
        print(f"\n{token}\n")
        return 0


async def _list() -> int:
    async with get_sessionmaker()() as db:
        users = (await db.execute(select(User).order_by(User.created_at))).scalars().all()
    if not users:
        print("no users yet. The first one is created by scripts/install.sh, or with --create here.")
        return 0
    for u in users:
        print(f"{u.role:<10} {u.name}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="the user to mint a token for")
    p.add_argument("--role", default="annotator", choices=("annotator", "reviewer", "admin"),
                   help="role to use when --create makes a new user")
    p.add_argument("--create", action="store_true", help="create the user if they do not exist")
    p.add_argument("--revoke-existing", action="store_true",
                   help="invalidate every token already issued to this user before minting the new one")
    p.add_argument("--list", action="store_true", help="list users and exit")
    args = p.parse_args()

    if args.list:
        raise SystemExit(asyncio.run(_list()))
    if not args.name:
        p.error("--name is required (or use --list)")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
