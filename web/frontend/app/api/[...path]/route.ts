// Proxy from the Vercel-hosted UI to the reduction worker.
//
// The worker cannot run on Vercel: a reduction runs the user's test
// command thousands of times over minutes to hours, needs a real Python
// environment with their dependencies installed, and keeps state on
// disk. Vercel functions cap out well below that. So the UI lives here
// and the engine lives on a machine that can hold a long job.
//
// Requests pass through this route rather than going straight to the
// worker so the browser only ever talks to one origin, and so
// WORKER_URL stays a server-side secret.

const WORKER_URL = process.env.WORKER_URL ?? 'http://127.0.0.1:8000';

// Every handler is the same shape: rewrite the path, forward the body,
// hand back whatever comes out. Streaming the response through means a
// result download is not buffered in the function's memory.
async function forward(request: Request, path: string[]) {
  const target = new URL(request.url);
  const url = `${WORKER_URL}/api/${path.join('/')}${target.search}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: request.method,
      headers: passthroughHeaders(request.headers),
      body: request.method === 'GET' ? undefined : request.body,
      // Required by undici whenever a streamed body is forwarded.
      // @ts-expect-error -- not in the DOM lib types, valid in Node.
      duplex: 'half',
      cache: 'no-store',
    });
  } catch {
    // A worker that is down is the single most likely failure here, and
    // "fetch failed" tells a user nothing they can act on.
    return Response.json(
      {
        detail:
          'The reduction worker is unreachable. If you are running ' +
          'locally, start it with `uvicorn main:app` in web/worker; if ' +
          'this is deployed, check WORKER_URL and that the service is up.',
      },
      { status: 502 },
    );
  }

  return new Response(response.body, {
    status: response.status,
    headers: {
      'content-type':
        response.headers.get('content-type') ?? 'application/json',
      ...(response.headers.get('content-disposition')
        ? {
            'content-disposition': response.headers.get(
              'content-disposition',
            ) as string,
          }
        : {}),
    },
  });
}

function passthroughHeaders(headers: Headers) {
  const out = new Headers();
  // content-type carries the multipart boundary, without which the
  // worker cannot parse an upload. Everything else (host, connection,
  // content-length) is about this hop and must not be copied.
  const contentType = headers.get('content-type');
  if (contentType) out.set('content-type', contentType);
  return out;
}

export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  return forward(request, (await context.params).path);
}

export async function POST(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  return forward(request, (await context.params).path);
}

// Uploads and long polls both need the Node runtime, not the edge one.
export const runtime = 'nodejs';
export const maxDuration = 60;
