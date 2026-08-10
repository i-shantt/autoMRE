/** @type {import('next').NextConfig} */
const nextConfig = {
  // Uploads go through a route handler to the worker. Next's default
  // body limit is far below a project archive, and the limit that
  // actually matters is enforced by the worker (MAX_ARCHIVE_BYTES), so
  // this only has to be big enough not to reject first.
  experimental: {
    serverActions: { bodySizeLimit: '30mb' },
  },
};

module.exports = nextConfig;
