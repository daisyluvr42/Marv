import { ImageResponse } from 'next/og';

export const size = {
  width: 512,
  height: 512
};

export const contentType = 'image/png';

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background:
            'radial-gradient(circle at 30% 20%, #3db2ff 0%, #146eb4 50%, #0b141e 100%)',
          color: '#ffffff',
          fontSize: 220,
          fontWeight: 700
        }}
      >
        M
      </div>
    ),
    { ...size }
  );
}
