// Browser-side indexed MCAP reader for the Session Inspector. The native panels read a session's MCAP
// straight from the object store over presigned HTTP range requests (MCAP is built for indexed random
// access), so no server streams full files. One ts_ns time base: message log times are the absolute clock.
//
// ts_ns values (~1.8e18) exceed Number.MAX_SAFE_INTEGER, so absolute times stay bigint here; panels convert
// to relative seconds for rendering.

import { McapIndexedReader, type McapTypes } from "@mcap/core";
import { decompress as zstdDecompress } from "fzstd";
import * as lz4 from "lz4js";

// Real MCAP compresses its chunks (zstd is the common default, lz4 also seen). Pure-JS decompressors keep
// the bundle free of the wasm-module plumbing webpack cannot resolve. bz2 is not used by MCAP recorders.
const DECOMPRESS_HANDLERS: Record<string, (data: Uint8Array, size: bigint) => Uint8Array> = {
  zstd: (data) => zstdDecompress(data),
  lz4: (data) => new Uint8Array(lz4.decompress(data)),
};

// A random-access reader over a presigned URL using Range requests, satisfying @mcap/core's IReadable.
class RangeReadable {
  private constructor(private url: string, private _size: bigint) {}

  static async open(url: string): Promise<RangeReadable> {
    // A presigned URL is signed for GET, so a HEAD request is rejected. Read the total size from a 1-byte
    // range GET's Content-Range header (bytes 0-0/<total>); if the server ignores Range and returns the
    // whole body, fall back to its length.
    const r = await fetch(url, { headers: { Range: "bytes=0-0" } });
    if (r.status !== 206 && r.status !== 200) throw new Error(`could not open MCAP: HTTP ${r.status}`);
    let size = 0n;
    const cr = r.headers.get("content-range");
    if (cr && cr.includes("/")) size = BigInt(cr.split("/")[1]);
    else size = BigInt((await r.arrayBuffer()).byteLength);
    if (size === 0n) throw new Error("could not determine MCAP size");
    return new RangeReadable(url, size);
  }

  async size(): Promise<bigint> {
    return this._size;
  }

  async read(offset: bigint, length: bigint): Promise<Uint8Array> {
    const start = offset;
    const end = offset + length - 1n;
    const r = await fetch(this.url, { headers: { Range: `bytes=${start}-${end}` } });
    if (r.status !== 206 && r.status !== 200) throw new Error(`range read failed: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }
}

export type TopicInfo = { topic: string; schema: string; encoding: string; channelId: number };
export type DecodedMessage = { topic: string; logTime: bigint; value: unknown; raw: Uint8Array };

export class SessionMcap {
  private topicToChannel = new Map<string, McapTypes.Channel>();
  private constructor(public reader: McapIndexedReader) {
    for (const ch of reader.channelsById.values()) this.topicToChannel.set(ch.topic, ch);
  }

  static async open(url: string): Promise<SessionMcap> {
    const readable = await RangeReadable.open(url);
    const reader = await McapIndexedReader.Initialize({ readable, decompressHandlers: DECOMPRESS_HANDLERS });
    return new SessionMcap(reader);
  }

  topics(): TopicInfo[] {
    const out: TopicInfo[] = [];
    for (const ch of this.reader.channelsById.values()) {
      const schema = ch.schemaId ? this.reader.schemasById.get(ch.schemaId) : undefined;
      out.push({ topic: ch.topic, schema: schema?.name ?? "", encoding: ch.messageEncoding, channelId: ch.id });
    }
    return out.sort((a, b) => a.topic.localeCompare(b.topic));
  }

  timeRange(): [bigint, bigint] | null {
    const s = this.reader.statistics;
    if (s && s.messageStartTime > 0n) return [s.messageStartTime, s.messageEndTime];
    return null;
  }

  private decode(channel: McapTypes.Channel, data: Uint8Array): unknown {
    // JSON-encoded channels decode directly (the platform's fixtures and any JSON MCAP). Non-JSON payloads
    // (protobuf/cdr) are handed back raw; the image panel uses the extracted-frames fast path and the raw
    // panel shows bytes, while Lichtblick covers full protobuf inspection.
    if (channel.messageEncoding === "json") {
      try {
        return JSON.parse(new TextDecoder().decode(data));
      } catch {
        return null;
      }
    }
    return null;
  }

  // Stream decoded messages for the given topics within [startTime, endTime] (absolute ts_ns, bigint).
  async *messages(topics: string[], startTime?: bigint, endTime?: bigint): AsyncGenerator<DecodedMessage> {
    for await (const m of this.reader.readMessages({ topics, startTime, endTime })) {
      const ch = this.reader.channelsById.get(m.channelId);
      if (!ch) continue;
      yield { topic: ch.topic, logTime: m.logTime, value: this.decode(ch, m.data), raw: m.data };
    }
  }

  // The single message on a topic at or just before ts (for panels that show "the value now").
  async latestAt(topic: string, ts: bigint, lookbackNs = 2_000_000_000n): Promise<DecodedMessage | null> {
    let best: DecodedMessage | null = null;
    for await (const m of this.messages([topic], ts > lookbackNs ? ts - lookbackNs : 0n, ts)) best = m;
    return best;
  }
}
