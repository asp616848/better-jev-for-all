/**
 * Real, runnable usage examples for ekvachan-client (TypeScript).
 *
 * Run against a locally hosted ekVachan server (`uvicorn serve.server:app`)
 * with `npx tsx examples/basic-usage.ts` or after `npm run build`, with
 * `node --experimental-strip-types examples/basic-usage.ts` (Node 22+) —
 * adjust to your runner of choice. Requires Node >= 18 (global `fetch`).
 */

import { EkVachanClient, EkVachanAPIError, type Question } from "../src/index.js";

async function main() {
  const client = new EkVachanClient({ baseUrl: "http://localhost:8000" });

  // 1. A request that works today: `choice` over the one schema the
  //    Phase 1 checkpoint was actually trained on.
  const nliResult = await client.choice({
    state:
      "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.",
    options: ["entailment", "neutral", "contradiction"],
  });
  console.log("nli result:", nliResult);
  // -> { choice: "entailment", probabilities: { entailment: 0.98, ... }, confidence: 0.98 }

  // 2. The full request shape, including a question that mirrors
  //    PRD.md Section 1.2's own example — but WILL 501 today, since no
  //    model is trained for an arbitrary option set like this one.
  const questions: Record<string, Question> = {
    nli: { type: "choice", options: ["entailment", "neutral", "contradiction"] },
    category: {
      type: "choice",
      instructions: "Classify the customer's issue.",
      options: ["billing", "technical", "other"],
    },
  };

  try {
    const response = await client.systemOne({
      state: "A customer emails asking why their invoice doubled this month.",
      questions,
    });
    console.log("full response:", response);
  } catch (err) {
    if (err instanceof EkVachanAPIError) {
      // Expected today: 501, because "category"'s options aren't the
      // trained schema. See sdk/typescript/README.md.
      console.log(`call failed as expected: ${err.statusCode} ${err.message}`);
    } else {
      throw err;
    }
  }

  // 3. Health check.
  console.log("health:", await client.health());
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
