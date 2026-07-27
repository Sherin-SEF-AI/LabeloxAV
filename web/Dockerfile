# LabeloxAV web (Next.js) production image: build once, serve with `next start`.
# Build:  docker build -t labeloxav-web ./web
# The API base is same-origin via the rewrite in next.config; set LBX_API_URL to point the proxy at the API
# service (e.g. http://api:8000) in the compose network.

FROM node:20-slim AS build
WORKDIR /web
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

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
