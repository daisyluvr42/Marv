import { ImageResponse } from 'next/og';

export const size = {
  width: 180,
  height: 180
};

export const contentType = 'image/png';

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 36,
          background:
            'radial-gradient(circle at 30% 20%, #3db2ff 0%, #146eb4 50%, #0b141e 100%)',
          color: '#ffffff',
          fontSize: 92,
          fontWeight: 700
        }}
      >
        M
      </div>
    ),
    { ...size }
  );
}
