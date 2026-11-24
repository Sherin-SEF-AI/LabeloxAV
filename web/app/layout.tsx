import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import Onboarding from "@/components/shell/Onboarding";
import GlobalLoadingBar from "@/components/GlobalLoadingBar";
import AuthBootstrap from "@/components/AuthBootstrap";
import Toaster from "@/components/Toaster";
import ConfirmProvider from "@/components/ConfirmProvider";
import ConsoleModal from "@/components/console/ConsoleModal";
import ThemeSync from "@/components/ThemeSync";
import { BOOT_SCRIPT } from "@/lib/bootScript";

const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });

export const metadata: Metadata = {
  title: "LabeloxAV",
  description: "India autonomous-driving data engine: review workstation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // lang and data-theme are a person's stored choices, which the server cannot know. BOOT_SCRIPT sets
    // both in <head>, before the body is parsed, so the first paint is in the right palette and the
    // document announces the right language. suppressHydrationWarning is for exactly those two attributes:
    // it covers this element only, not its children.
    <html lang="en" suppressHydrationWarning
      className={`${display.variable} ${GeistSans.variable} ${GeistMono.variable}`}
      style={{ "--font-body": "var(--font-geist-sans)", "--font-mono": "var(--font-geist-mono)" } as React.CSSProperties}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: BOOT_SCRIPT }} />
      </head>
      <body className="font-body bg-bg text-ink antialiased">
        <ThemeSync />
        <GlobalLoadingBar />
        <AuthBootstrap />
        <Toaster />
        <ConfirmProvider>{children}</ConfirmProvider>
        {/* Shown once per person, only when signed in. Mounted here rather than per page so it does not
            depend on which page they happen to land on first. */}
        <Onboarding />
        {/* The console opens over whatever raised the question, so it is mounted once here rather than per
            page: the frame editor is exactly where somebody asks why something is slow, and that is the one
            page they cannot afford to navigate away from. */}
        <ConsoleModal />
      </body>
    </html>
  );
}
