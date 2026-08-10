import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'autoMRE — shrink a bug to its smallest reproduction',
  description:
    'Upload a Python project and a failing-or-passing test command. ' +
    'autoMRE deletes everything the test does not need and gives you ' +
    'back the smallest project that still behaves the same way.',
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
