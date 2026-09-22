# ekvachan-client (TypeScript)

A minimal, fetch-based TypeScript client for ekVachan's `POST /v1/systemone`
endpoint (`serve/server.py`). No framework, no heavy dependency — just the
global `fetch` (Node >= 18, or any modern browser).

**Type-checked and compiled, but not runtime-executed.** This package was
written against `serve/server.py`'s actual Pydantic models and response
shape. `npm install && npx tsc -p tsconfig.json --noEmit` (strict mode)
and a full `npm run build` both pass cleanly on `src/client.ts`,
`src/index.ts`, and `examples/basic-usage.ts` — verified in this
environment (Node v26, TypeScript 5.9). What was **not** verified: an
actual HTTP round-trip against a running server (no test harness/mock
server was built for the TS client, unlike the Python client's
execution-verified test suite), and `npm publish`-level packaging.

## Honest limitation, right up front

The Phase 1 reference server only returns a real answer for **one**
question shape:

```ts
{ type: "choice", options: ["entailment", "neutral", "contradiction"] }
```

Any other `choice` options array, or any `score`/`noul` question,
currently gets **HTTP 501** — there's no trained model for those yet
(PRD.md Section 10a / 13a.3). This client types and sends the full
`choice`/`score`/`noul` wire shape (that's the target contract the
server's Pydantic models already accept), but the example below that
would 501 today is labeled as such rather than presented as if it works.

## Install

Not yet published to npm. Use it locally from this repo, e.g. via a
workspace/path dependency:

```json
{ "dependencies": { "ekvachan-client": "file:../path/to/sdk/typescript" } }
```

Then build it once:

```bash
cd sdk/typescript
npm install
npm run build
```

## Usage

### 1. A request that works today (`choice` over the trained schema)

```ts
import { EkVachanClient } from "ekvachan-client";

const client = new EkVachanClient({ baseUrl: "http://localhost:8000" });

const result = await client.choice({
  state: "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.",
  options: ["entailment", "neutral", "contradiction"],
});
console.log(result);
// { choice: "entailment", probabilities: { entailment: 0.98, ... }, confidence: 0.98 }
```

### 2. The full request shape, including a question that will 501 today

```ts
import { EkVachanClient, EkVachanAPIError } from "ekvachan-client";

const client = new EkVachanClient({ baseUrl: "http://localhost:8000" });

try {
  const response = await client.systemOne({
    state: "A customer emails asking why their invoice doubled this month.",
    questions: {
      // This shape works against the Phase 1 checkpoint:
      nli: { type: "choice", options: ["entailment", "neutral", "contradiction"] },
      // This mirrors PRD.md Section 1.2's own example request, but WILL
      // throw EkVachanAPIError with statusCode 501 today -- no model is
      // trained for an arbitrary option set like this one.
      category: {
        type: "choice",
        instructions: "Classify the customer's issue.",
        options: ["billing", "technical", "other"],
      },
    },
  });
  console.log(response);
} catch (err) {
  if (err instanceof EkVachanAPIError) {
    console.log(`call failed: ${err.statusCode} ${err.message}`);
    // -> call failed: 501 question 'category': this checkpoint only supports
    //    options==['entailment', 'neutral', 'contradiction'] (trained schema); ...
  } else {
    throw err;
  }
}
```

### 3. Health check

```ts
await client.health(); // { status: "ok" }
```

See `examples/basic-usage.ts` for a runnable version of both examples
above.

## Verification status

TypeScript source was hand-written against `serve/server.py`'s actual
`SystemOneRequest`/`Question` Pydantic models and response dict (not
guessed from the PRD's aspirational description), then compiled and
strict-mode type-checked (`npm install`, `npx tsc --noEmit`, `npm run
build`) in this environment — all pass cleanly, including the
`examples/basic-usage.ts` file. What was **not** verified: an actual
network call against a running `serve/server.py` instance (no
mock-server test harness was built for this client, unlike the Python
client's `tests/test_client.py`), and `npm publish`-level packaging
(package.json fields are correct by inspection, not by a real publish
dry-run).
