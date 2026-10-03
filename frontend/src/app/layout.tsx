import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PDF Q&A — Intelligent Document Assistant',
  description: 'Upload PDFs and ask questions with AI-powered answers grounded in your documents.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
