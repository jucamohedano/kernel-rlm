/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/kernel-rlm',
  trailingSlash: true,
  images: { unoptimized: true },
};

module.exports = nextConfig;
