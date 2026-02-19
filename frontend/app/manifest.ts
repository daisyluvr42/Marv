import type { MetadataRoute } from 'next';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Marv Console',
    short_name: 'Marv',
    description: 'Local-first agent runtime console (MacOS + iOS friendly).',
    start_url: '/chat',
    display: 'standalone',
    background_color: '#0b141e',
    theme_color: '#0b141e',
    orientation: 'portrait',
    icons: [
      {
        src: '/icon?size=192',
        sizes: '192x192',
        type: 'image/png'
      },
      {
        src: '/icon?size=512',
        sizes: '512x512',
        type: 'image/png'
      },
      {
        src: '/apple-icon',
        sizes: '180x180',
        type: 'image/png'
      }
    ]
  };
}
