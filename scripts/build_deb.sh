#!/usr/bin/env bash
#
# Build the Debian package: labeloxav_<version>_all.deb in dist/.
#
#   ./scripts/build_deb.sh            build
#   ./scripts/build_deb.sh --check    build, then verify the archive and its contents
#
# The payload is the committed tree at HEAD, not the working directory. A release is a commit, and a
# package built from whatever happened to be on disk would be unreproducible in exactly the way that
# matters: nobody could rebuild it and get the same bytes. The packaging files themselves are read from
# the working tree so they can be tested before they are committed.
#
# What is left out of /opt/labeloxav, and why: the documentation site and its films (100 MB of video
# a server never serves), model weights (downloaded on demand by scripts/download_models.py; the
# tracked copies are LFS pointers, which are useless), tests, demos, CI configuration, and this
# packaging directory. Everything the Docker build context needs is kept, because the containers are
# built from this tree.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
[ -n "$VERSION" ] || { echo "could not read the version from pyproject.toml" >&2; exit 1; }

PKG="labeloxav_${VERSION}_all"
BUILD="$ROOT/.build/deb"
STAGE="$BUILD/$PKG"
OUT="$ROOT/dist/$PKG.deb"

rm -rf "$STAGE"
mkdir -p "$STAGE/opt/labeloxav" "$STAGE/usr/bin" "$STAGE/usr/share/doc/labeloxav" "$STAGE/DEBIAN" "$ROOT/dist"

echo "==> payload: git archive HEAD ($(git rev-parse --short HEAD))"
git archive --format=tar HEAD | tar -x -C "$STAGE/opt/labeloxav"

# Pruned after extraction rather than filtered on the way in, so the exclusion list is one plain list
# of paths that anyone can read, and so a directory that was renamed shows up as a missing rm target.
for p in docs models tests demos site .github packaging reports mkdocs.yml AUDIT.md; do
  rm -rf "$STAGE/opt/labeloxav/$p"
done
# Keep the licence beside the code; dpkg convention also wants a copy under /usr/share/doc.
cp LICENSE "$STAGE/usr/share/doc/labeloxav/LICENSE" 2>/dev/null || true
cp README.md "$STAGE/usr/share/doc/labeloxav/README.md"
cp packaging/deb/copyright "$STAGE/usr/share/doc/labeloxav/copyright"

install -m 0755 packaging/deb/labeloxav "$STAGE/usr/bin/labeloxav"
install -m 0755 packaging/deb/postinst "$STAGE/DEBIAN/postinst"
install -m 0755 packaging/deb/postrm "$STAGE/DEBIAN/postrm"

# Debian wants the changelog compressed at maximum level; dpkg does not require it, but a package that
# says which commit it was built from is one somebody can reason about later.
printf 'labeloxav (%s) unstable; urgency=medium\n\n  * Built from commit %s.\n\n -- Sherin Joseph Roy <sherin.srambickal@gmail.com>  %s\n' \
  "$VERSION" "$(git rev-parse HEAD)" "$(date -R)" | gzip -9n > "$STAGE/usr/share/doc/labeloxav/changelog.Debian.gz"

SIZE_KB="$(du -sk --exclude=DEBIAN "$STAGE" | cut -f1)"
sed -e "s/@VERSION@/$VERSION/" -e "s/@INSTALLED_SIZE@/$SIZE_KB/" packaging/deb/control.in > "$STAGE/DEBIAN/control"

# Files that only make sense on the machine that ran them.
find "$STAGE/opt/labeloxav" -name '__pycache__' -type d -prune -exec rm -rf {} +
chmod -R go-w "$STAGE/opt/labeloxav"

echo "==> dpkg-deb"
dpkg-deb --build --root-owner-group "$STAGE" "$OUT" >/dev/null
echo "==> wrote $OUT ($(du -h "$OUT" | cut -f1))"

if [ "${1:-}" = "--check" ]; then
  echo "==> check"
  dpkg-deb --info "$OUT" | sed 's/^/   /'
  # The listing is read once into a file and grepped from there. Piping `dpkg-deb --contents` into
  # `grep -q` looks equivalent and is not: grep exits on its first match and closes the pipe, tar dies
  # of SIGPIPE, and under pipefail that reads as the file being missing. The first run of this check
  # reported install.sh absent from a package that a clean container then installed and ran.
  listing="$BUILD/contents.txt"
  dpkg-deb --contents "$OUT" > "$listing"
  echo "   $(wc -l < "$listing") entries"
  for must in ./usr/bin/labeloxav ./opt/labeloxav/scripts/install.sh ./opt/labeloxav/docker-compose.yml \
              ./opt/labeloxav/docker-compose.app.yml ./opt/labeloxav/Dockerfile ./opt/labeloxav/pyproject.toml \
              ./opt/labeloxav/web/package.json ./opt/labeloxav/ontology ./usr/share/doc/labeloxav/copyright; do
    grep -q " $must" "$listing" || { echo "   MISSING: $must" >&2; exit 1; }
  done
  # `.env$` is anchored: `.env.example` ships on purpose, the generated `.env` must never.
  for mustnot in ./opt/labeloxav/docs/ ./opt/labeloxav/models/ ./opt/labeloxav/tests/ './opt/labeloxav/.env$' \
                 ./opt/labeloxav/packaging/ '\.mp4$' '\.pt$'; do
    if grep -qE " $mustnot" "$listing"; then echo "   MUST NOT SHIP: $mustnot" >&2; exit 1; fi
  done
  echo "   required files present, excluded files absent"
fi
