import type { Metadata } from 'next';

import { Sidebar } from '@/components/sidebar';

import './globals.css';

export const metadata: Metadata = {
  title: 'Blackbox Console',
  description: 'Agent runtime console'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className="mx-auto flex min-h-screen max-w-[1200px] flex-col md:flex-row">
          <Sidebar />
          <main className="flex-1 p-4 md:p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
