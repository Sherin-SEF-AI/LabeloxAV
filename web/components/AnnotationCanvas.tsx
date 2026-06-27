"use client";

import { useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KImage, Rect, Line, Circle } from "react-konva";

type Props = {
  imageUrl: string;
  imgWidth: number;
  imgHeight: number;
  bbox: number[]; // xyxy in image px
  maskPolygons: number[][]; // existing mask, flattened [x,y,...] per polygon
  candidatePolygons: number[][]; // SAM proposal
  clickPoint: number[] | null;
  onPointClick: (x: number, y: number) => void; // image-space coords
};

export default function AnnotationCanvas({
  imageUrl,
  imgWidth,
  imgHeight,
  bbox,
  maskPolygons,
  candidatePolygons,
  clickPoint,
  onPointClick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const im = new window.Image();
    im.crossOrigin = "anonymous";
    im.src = imageUrl;
    im.onload = () => setImg(im);
  }, [imageUrl]);

  useEffect(() => {
    const fit = () => {
      const w = containerRef.current?.clientWidth ?? imgWidth;
      setScale(Math.min(1, w / imgWidth));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [imgWidth]);

  const sx = (pts: number[]) => pts.map((v, i) => (i % 2 === 0 ? v * scale : v * scale));

  return (
    <div ref={containerRef} className="w-full">
      <Stage
        width={imgWidth * scale}
        height={imgHeight * scale}
        onClick={(e) => {
          const stage = e.target.getStage();
          const p = stage?.getPointerPosition();
          if (p) onPointClick(p.x / scale, p.y / scale);
        }}
        className="cursor-crosshair"
      >
        <Layer>
          {img && <KImage image={img} width={imgWidth * scale} height={imgHeight * scale} />}
          {/* proposed box */}
          {bbox.length === 4 && (
            <Rect
              x={bbox[0] * scale}
              y={bbox[1] * scale}
              width={(bbox[2] - bbox[0]) * scale}
              height={(bbox[3] - bbox[1]) * scale}
              stroke="#FF7A2F"
              strokeWidth={1.5}
              dash={[6, 4]}
            />
          )}
          {/* existing mask */}
          {maskPolygons.map((poly, i) => (
            <Line key={`m${i}`} points={sx(poly)} closed stroke="#58A6FF" strokeWidth={1.5} fill="rgba(88,166,255,0.18)" />
          ))}
          {/* SAM candidate */}
          {candidatePolygons.map((poly, i) => (
            <Line key={`c${i}`} points={sx(poly)} closed stroke="#56D364" strokeWidth={2} fill="rgba(86,211,100,0.25)" />
          ))}
          {clickPoint && <Circle x={clickPoint[0] * scale} y={clickPoint[1] * scale} radius={4} fill="#FF7A2F" />}
        </Layer>
      </Stage>
    </div>
  );
}
