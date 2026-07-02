// Minimal typing for the pure-JS lz4 frame decompressor used by the MCAP reader (no bundled types).
declare module "lz4js" {
  export function decompress(data: Uint8Array): Uint8Array;
  export function compress(data: Uint8Array): Uint8Array;
}
