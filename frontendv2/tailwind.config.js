/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // ── M3 EXPRESSIVE: FONT FAMILIES ──────────────────────────
      fontFamily: {
        headline: ['Google Sans Display', 'Google Sans', 'sans-serif'],
        body: ['Roboto Flex', 'Roboto', 'Google Sans', 'sans-serif'],
      },

      // ── M3 EXPRESSIVE: COLOR PALETTE ──────────────────────────
      colors: {
        m3: {
          // Surface hierarchy (light theme)
          'surface': '#FDFBFF',
          'surface-dim': '#DDD8E1',
          'surface-bright': '#FDFBFF',
          'surface-container-lowest': '#FFFFFF',
          'surface-container-low': '#F0EDF4',
          'surface-container': '#E9E5EE',
          'surface-container-high': '#E3DFE8',
          'surface-container-highest': '#DDD8E2',

          // Primary (Action colour — Google Blue)
          'primary': '#0B57D0',
          'on-primary': '#FFFFFF',
          'primary-container': '#D3E3FD',
          'on-primary-container': '#041E49',

          // Secondary (Tonal highlight)
          'secondary': '#005A99',
          'on-secondary': '#FFFFFF',
          'secondary-container': '#C2E7FF',
          'on-secondary-container': '#001D35',

          // Tertiary (Accent — Violet tint for M3 Expressive)
          'tertiary': '#6750A4',
          'on-tertiary': '#FFFFFF',
          'tertiary-container': '#EADDFF',
          'on-tertiary-container': '#21005D',

          // Error / Status
          'error': '#B3261E',
          'on-error': '#FFFFFF',
          'error-container': '#FFEDD5',
          'on-error-container': '#92400E',

          // Success (Custom token)
          'success': '#0F9D58',
          'success-container': '#C4EED0',
          'on-success-container': '#0F5223',

          // Warning (amber)
          'warning': '#D97706',
          'warning-container': '#FEF7E0',
          'on-warning-container': '#78350F',

          // Text / On-surface
          'on-surface': '#1B1B1F',
          'on-surface-variant': '#44474F',
          'outline': '#747775',
          'outline-variant': '#C4C7C5',

          // Inverse (dark surface)
          'inverse-surface': '#313033',
          'inverse-on-surface': '#F4EFF4',
          'inverse-primary': '#AAC7FF',

          // Scrim
          'scrim': '#000000',
        }
      },

      // ── M3 EXPRESSIVE: BORDER RADII ────────────────────────────
      borderRadius: {
        'xs': '4px',
        'sm': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px',
        '2xl': '28px',
        '3xl': '36px',
        'full': '9999px',
      },

      // ── M3 EXPRESSIVE: SHADOWS (Elevation) ─────────────────────
      boxShadow: {
        'elevation-1': '0 1px 2px rgba(0,0,0,0.12), 0 1px 3px rgba(0,0,0,0.08)',
        'elevation-2': '0 2px 6px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08)',
        'elevation-3': '0 4px 12px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08)',
        'elevation-4': '0 8px 24px rgba(0,0,0,0.12), 0 6px 12px rgba(0,0,0,0.08)',
        'glow-primary': '0 0 24px rgba(11,87,208,0.25)',
        'glow-tertiary': '0 0 24px rgba(103,80,164,0.20)',
      },

      // ── M3 EXPRESSIVE: TYPOGRAPHY SCALE ────────────────────────
      fontSize: {
        'display-large': ['57px', { lineHeight: '64px', letterSpacing: '-0.25px', fontWeight: '400' }],
        'display-medium': ['45px', { lineHeight: '52px', letterSpacing: '0px', fontWeight: '400' }],
        'display-small': ['36px', { lineHeight: '44px', letterSpacing: '0px', fontWeight: '400' }],
        'headline-large': ['32px', { lineHeight: '40px', letterSpacing: '0px', fontWeight: '400' }],
        'headline-medium': ['28px', { lineHeight: '36px', letterSpacing: '0px', fontWeight: '400' }],
        'headline-small': ['24px', { lineHeight: '32px', letterSpacing: '0px', fontWeight: '400' }],
        'title-large': ['22px', { lineHeight: '28px', letterSpacing: '0px', fontWeight: '500' }],
        'title-medium': ['16px', { lineHeight: '24px', letterSpacing: '0.15px', fontWeight: '500' }],
        'title-small': ['14px', { lineHeight: '20px', letterSpacing: '0.1px', fontWeight: '500' }],
        'body-large': ['16px', { lineHeight: '24px', letterSpacing: '0.5px', fontWeight: '400' }],
        'body-medium': ['14px', { lineHeight: '20px', letterSpacing: '0.25px', fontWeight: '400' }],
        'body-small': ['12px', { lineHeight: '16px', letterSpacing: '0.4px', fontWeight: '400' }],
        'label-large': ['14px', { lineHeight: '20px', letterSpacing: '0.1px', fontWeight: '500' }],
        'label-medium': ['12px', { lineHeight: '16px', letterSpacing: '0.5px', fontWeight: '500' }],
        'label-small': ['11px', { lineHeight: '16px', letterSpacing: '0.5px', fontWeight: '500' }],
      },

      // ── ANIMATION DURATIONS ─────────────────────────────────────
      transitionDuration: {
        'short1': '50ms',
        'short2': '100ms',
        'short3': '150ms',
        'short4': '200ms',
        'medium1': '250ms',
        'medium2': '300ms',
        'medium3': '350ms',
        'medium4': '400ms',
        'long1': '450ms',
        'long2': '500ms',
      },
    },
  },
  plugins: [],
}