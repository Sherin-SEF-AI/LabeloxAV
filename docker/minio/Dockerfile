# MinIO, built from source at a pinned release.
#
# MinIO stopped publishing its server images: quay.io/minio/minio answers 401 to an anonymous pull and the
# Docker Hub repository is gone, so a fresh install (and CI) failed at the first `docker compose up`. Every
# machine that worked had the image cached from before. The free edition is now distributed as source only,
# and the source of this exact release stays available from GitHub and the Go module mirror, so building it
# here is the copy that cannot disappear.
#
# Same release as the image this replaces, same /data layout and ports, so existing volumes keep working.
# Bump MINIO_RELEASE deliberately, with its release notes read: it is the object store holding every frame.
FROM golang:1.22-bookworm AS build
ARG MINIO_RELEASE=RELEASE.2024-10-13T13-34-11Z
RUN git clone --depth 1 --branch "${MINIO_RELEASE}" https://github.com/minio/minio.git /src
WORKDIR /src
# Static, stripped, with the release recorded in the binary so `minio --version` names what is running.
RUN CGO_ENABLED=0 go build -trimpath \
      -ldflags "-s -w -X github.com/minio/minio/cmd.Version=${MINIO_RELEASE#RELEASE.} -X github.com/minio/minio/cmd.ReleaseTag=${MINIO_RELEASE} -X github.com/minio/minio/cmd.CommitID=$(git rev-parse HEAD)" \
      -o /out/minio .

FROM debian:bookworm-slim
# curl for the compose healthcheck (GET /minio/health/live); ca-certificates for TLS to anything upstream.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/minio /usr/bin/minio
EXPOSE 9000 9001
VOLUME ["/data"]
ENTRYPOINT ["/usr/bin/minio"]
