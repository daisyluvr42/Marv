import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0B0F14',
        surface: '#121826',
        surface2: '#182235',
        border: '#243047',
        text: '#E6EDF3',
        muted: '#9FB0C3',
        primary: '#4F8CFF',
        success: '#2DD4BF',
        warning: '#FBBF24',
        danger: '#F87171'
      }
    }
  },
  plugins: []
};

export default config;
