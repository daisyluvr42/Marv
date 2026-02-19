'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Blocks, CheckSquare, Gauge, Hammer, MessageSquare, MemoryStick, Settings2 } from 'lucide-react';

const nav = [
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/tasks', label: 'Tasks', icon: CheckSquare },
  { href: '/approvals', label: 'Approvals', icon: Hammer },
  { href: '/memory', label: 'Memory', icon: MemoryStick },
  { href: '/config', label: 'Config', icon: Settings2 },
  { href: '/tools', label: 'Tools', icon: Blocks }
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-full border-b border-border bg-surface p-4 md:h-screen md:w-64 md:border-b-0 md:border-r">
      <div className="mb-6 flex items-center gap-2 text-text">
        <Gauge className="h-5 w-5 text-primary" />
        <span className="font-semibold">Blackbox Console</span>
      </div>
      <nav className="grid grid-cols-2 gap-2 md:grid-cols-1">
        {nav.map((item) => {
          const Icon = item.icon;
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition ${
                active
                  ? 'border-primary bg-primary/15 text-text'
                  : 'border-border bg-surface2 text-muted hover:border-primary/50 hover:text-text'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
