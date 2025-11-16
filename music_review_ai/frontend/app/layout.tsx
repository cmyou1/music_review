import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Music Review AI",
  description: "Upload tracks, get AI reviews, recommendations, and meta-reviews.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body className="bg-slate-950 text-white min-h-screen">{children}</body>
    </html>
  );
}
