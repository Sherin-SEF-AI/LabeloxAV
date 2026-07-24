"""LabeloxAV command line, over the same REST API the web app uses.

A thin wrapper on sdk.labelox_client, so the CLI cannot drift from the client or the server. Reads the server
url and token from LBX_URL / LBX_TOKEN (or LBX_USER_ID on a dev server with auth disabled).

    python -m sdk.cli health
    python -m sdk.cli projects
    python -m sdk.cli facets --predicate '{"states":["accepted"]}'
    python -m sdk.cli tag --predicate '{"weather":["rain"]}' --add night_audit
    python -m sdk.cli export --name my-slice --formats coco,cvat
"""

from __future__ import annotations

import json
import os
import sys

import click

from sdk.labelox_client import Labelox, LabeloxError


def _client() -> Labelox:
    return Labelox(os.environ.get("LBX_URL", "http://localhost:8000"),
                   token=os.environ.get("LBX_TOKEN"),
                   user_id=os.environ.get("LBX_USER_ID"))


def _out(obj) -> None:
    click.echo(json.dumps(obj, indent=2, default=str))


@click.group()
def cli() -> None:
    """LabeloxAV CLI."""


@cli.command()
def health() -> None:
    """Server and dependency health."""
    _out(_client().health())


@cli.command()
def metrics() -> None:
    """Corpus counts and the auto-accept switch."""
    _out(_client().metrics())


@cli.command()
def projects() -> None:
    """List labeling projects."""
    _out(_client().projects())


@cli.command()
@click.option("--name", required=True)
@click.option("--honeypot-frac", type=float, default=0.0)
@click.option("--gold-id", default=None)
def create_project(name: str, honeypot_frac: float, gold_id: str | None) -> None:
    """Create a labeling project."""
    _out(_client().create_project(name, honeypot_frac=honeypot_frac, gold_id=gold_id))


@cli.command()
@click.option("--predicate", default="{}", help="explorer predicate as JSON")
def facets(predicate: str) -> None:
    """Faceted counts under a predicate."""
    _out(_client().facets(json.loads(predicate)))


@cli.command()
@click.option("--predicate", default="{}")
@click.option("--level", default="object", type=click.Choice(["object", "frame"]))
@click.option("--limit", default=20, help="how many ids to print")
def select(predicate: str, level: str, limit: int) -> None:
    """Resolve a predicate to a count and a sample of ids."""
    _out(_client().select(json.loads(predicate), level=level, limit=limit))


@cli.command()
@click.option("--predicate", required=True, help="what to tag, as JSON")
@click.option("--add", multiple=True)
@click.option("--remove", multiple=True)
@click.option("--level", default="object", type=click.Choice(["object", "frame"]))
def tag(predicate: str, add: tuple[str, ...], remove: tuple[str, ...], level: str) -> None:
    """Bulk add or remove curation tags."""
    _out(_client().tag(json.loads(predicate), add=list(add), remove=list(remove), level=level))


@cli.command("import")
@click.option("--format", "fmt", required=True,
              help="coco|yolo|pascalvoc|cvat|labelstudio|openlabel|nuscenes|kitti|bdd|parquet|images|video|mcap")
@click.option("--source", "source_uri", required=True)
@click.option("--city", default=None)
def import_(fmt: str, source_uri: str, city: str | None) -> None:
    """Start an import job."""
    _out(_client().start_import(fmt, source_uri, **({"city": city} if city else {})))


@cli.command()
@click.option("--name", required=True)
@click.option("--formats", default="coco,parquet", help="comma separated")
@click.option("--states", default=None, help="comma separated object states")
def export(name: str, formats: str, states: str | None) -> None:
    """Seal and export a dataset slice."""
    kw = {}
    if states:
        kw["states"] = states.split(",")
    _out(_client().export(name, formats=formats.split(","), **kw))


@cli.command()
@click.option("--project-id", default=None)
def scorecards(project_id: str | None) -> None:
    """Per-annotator throughput and honeypot quality."""
    _out(_client().scorecards(project_id))


@cli.command()
@click.option("--url", required=True)
@click.option("--event", "events", multiple=True, help="repeatable; omit to receive every event")
def add_webhook(url: str, events: tuple[str, ...]) -> None:
    """Register an outbound webhook. Prints the signing secret once."""
    _out(_client().create_webhook(url, list(events)))


def main() -> None:
    try:
        cli(standalone_mode=False)
    except LabeloxError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.Abort:
        sys.exit(130)


if __name__ == "__main__":
    main()
