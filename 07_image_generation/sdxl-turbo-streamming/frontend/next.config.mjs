import dotenv from 'dotenv';
dotenv.config();

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    BEAM_AUTH_BEARER: process.env.BEAM_AUTH_BEARER,
    BEAM_API_URL: process.env.BEAM_API_URL,
  },
};

export default nextConfig;
