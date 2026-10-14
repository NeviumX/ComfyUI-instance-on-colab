"""The processes the runtime starts of itself: python -m <this package> ACTION."""

import argparse
import asyncio
from pathlib import Path
import sys

from . import app_tree, commands, diagnosis, paths, proxy, service


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["proxy", "supervise", "diagnose", "app"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "app":
        if commands.drops_privileges():
            raise SystemExit("app actions must run as the application user")
        paths.BASE = Path(args.rest[0])
        app_tree.app_tree_action(args.rest[1], args.rest[2:], sys.stdin.buffer, sys.stdout.buffer)
    elif args.action == "proxy":
        from aiohttp import web
        web.run_app(proxy.create_proxy(service.load_config()), host="127.0.0.1", port=8189, access_log=None)
    elif args.action == "diagnose":
        asyncio.run(diagnosis.diagnose(service.load_config()))
    else:
        service.supervise(service.load_config())
