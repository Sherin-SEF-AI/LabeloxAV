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
  // konva's node build references the optional native 'canvas' package; the browser does not need
  // it. Alias it out so the bundle compiles (the canvas is client-only via dynamic ssr:false).
  webpack: (config) => {
    config.resolve.alias = { ...config.resolve.alias, canvas: false };
    return config;
  },
};

export default nextConfig;
