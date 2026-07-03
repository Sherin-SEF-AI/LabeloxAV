import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import GlobalLoadingBar from "@/components/GlobalLoadingBar";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });

export const metadata: Metadata = {
  title: "LabeloxAV",
  description: "India autonomous-driving data engine: review workstation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${GeistSans.variable} ${GeistMono.variable}`}
      style={{ "--font-body": "var(--font-geist-sans)", "--font-mono": "var(--font-geist-mono)" } as React.CSSProperties}>
      <body className="font-body bg-bg text-ink antialiased">
        <GlobalLoadingBar />
        {children}
      </body>
    </html>
  );
}
