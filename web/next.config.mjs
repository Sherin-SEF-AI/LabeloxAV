/** @type {import('next').NextConfig} */
const API = process.env.LBX_API_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
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
