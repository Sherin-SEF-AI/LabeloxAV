# LabeloxAV web (Next.js) production image: build once, serve with `next start`.
# Build:  docker build -t labeloxav-web ./web
# The API base is same-origin via the rewrite in next.config; set LBX_API_URL to point the proxy at the API
# service (e.g. http://api:8000) in the compose network.

FROM node:20-slim AS build
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
# Next compiles rewrite destinations into .next/routes-manifest.json during the build, so LBX_API_URL has to
# be present HERE. Setting it only at runtime leaves the manifest pointing at the build-time default
# (127.0.0.1:8000), and every proxied /api call from the browser fails with ECONNREFUSED even though the
# variable is visibly set in the running container. Defaults to the compose service name.
ARG LBX_API_URL=http://api:8000
ENV LBX_API_URL=${LBX_API_URL}
RUN npm run build
# `public/` is optional in Next and this app ships no static assets, so create it if absent: a bare COPY of a
# missing directory fails the build outright, which is a confusing way to learn there were no favicons.
RUN mkdir -p /web/public

FROM node:20-slim AS run
WORKDIR /web
ENV NODE_ENV=production
COPY --from=build /web/.next ./.next
COPY --from=build /web/public ./public
COPY --from=build /web/node_modules ./node_modules
COPY --from=build /web/package.json ./package.json
COPY --from=build /web/next.config.* ./
EXPOSE 3000
CMD ["npm", "run", "start"]
