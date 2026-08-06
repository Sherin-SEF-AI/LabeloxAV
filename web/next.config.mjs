/** @type {import('next').NextConfig} */
const API = process.env.LBX_API_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // react-konva + konva ship as ESM with a non-standard default-export shape; without transpiling them,
  // next/dynamic's convertModule can receive a bad module and throw "Cannot use 'in' operator to search
  // for 'default' in Layer" on a (StrictMode) re-mount. Transpiling them fixes the interop.
  transpilePackages: ["konva", "react-konva"],
  // Proxy /api to the FastAPI backend so the browser stays same-origin (image proxy + fetch).
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  experimental: {
    // The rewrite proxy defaults to a 30s ceiling, which is shorter than several endpoints legitimately
    // take. A promotion runs a full evaluation of both the challenger and the reigning champion against
    // gold, which on a cold cache is around 90s of GPU work; training, export and autolabel are the same
    // shape. At the default the browser is handed a bare 500 while the backend runs happily to completion,
    // so the gate would record a decision the operator was told had failed. That is the worst possible
    // reading for a governance control: it invites a retry of a promotion that already happened.
    //
    // This raises the dev ceiling above the slowest of those paths. It does not make the endpoints fast,
    // and a production ingress in front of the API needs its own matching timeout.
    proxyTimeout: 5 * 60 * 1000,
  },
  // konva's node build references the optional native 'canvas' package; the browser does not need
  // it. Alias it out so the bundle compiles (the canvas is client-only via dynamic ssr:false).
  webpack: (config) => {
    config.resolve.alias = { ...config.resolve.alias, canvas: false };
    return config;
  },
};

export default nextConfig;
