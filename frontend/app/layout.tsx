import type { Metadata, Viewport } from 'next';

import { Sidebar } from '@/components/sidebar';
import { PwaRegister } from '@/components/pwa-register';

import './globals.css';

export const metadata: Metadata = {
  title: 'Blackbox Console',
  description: 'Agent runtime console',
  manifest: '/manifest.webmanifest',
  applicationName: 'Marv Console',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'Marv Console'
  },
  formatDetection: {
    telephone: false
  }
};

export const viewport: Viewport = {
  themeColor: '#0b141e',
  viewportFit: 'cover'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <PwaRegister />
        <div className="mx-auto flex min-h-screen max-w-[1200px] flex-col md:flex-row">
          <Sidebar />
          <main className="flex-1 p-4 md:p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
