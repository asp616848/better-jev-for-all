/**
 * ekvachan-client — a minimal TypeScript client for ekVachan's
 * `/v1/systemone` endpoint.
 *
 * Mirrors the request/response shape implemented by `serve/server.py`
 * (see PRD.md Section 6.1/6.3 for the target contract this wire shape
 * follows). Fetch-based, no framework or heavy dependency.
 *
 * HONEST LIMITATION (Phase 1 reference server, as of 2026-09-22):
 * Only the `choice` primitive returns a real answer today, and only for
 * the exact option set `["entailment", "neutral", "contradiction"]` the
 * checkpoint was trained on. Any other `choice` options, or any
 * `score`/`noul` question, gets HTTP 501 from the server. This client
 * types and sends the full choice/score/noul wire shape (that's the
 * target contract the server's Pydantic models already accept), but
 * don't expect anything outside that one schema to succeed against the
 * current checkpoint. See PRD.md Section 10a / 13a.3.
 */

/** One typed question in a /v1/systemone request. Mirrors
 * `serve.server.Question` exactly: `type` is one of "choice", "score",
 * "noul" per the wire contract (PRD.md Section 1.2/6.1).
 *
 * Today's Phase 1 reference server only returns a real answer for
 * `type: "choice"` with `options` exactly
 * `["entailment", "neutral", "contradiction"]`; every other combination
 * currently raises an EkVachanAPIError with status 501.
 */
export interface Question {
  type: "choice" | "score" | "noul" | (string & {});
  instructions?: string;
  options?: string[];
  levels?: string[];
}

/** Request body for POST /v1/systemone, mirroring
 * `serve.server.SystemOneRequest` (`state`, `model`, `questions`). */
export interface SystemOneRequestBody {
  state: string;
  model?: string;
  questions: Record<string, Question>;
}

/** Result of a single `choice` question that succeeded. */
export interface ChoiceResult {
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

/** A single question's result. Only `choice` is populated by the
 * current server; this is typed generically since `score`/`noul` will
 * carry their own shapes once implemented (PRD.md Section 1.2). */
export type QuestionResult = ChoiceResult | Record<string, unknown>;

/** Parsed response from POST /v1/systemone. */
export interface SystemOneResponse {
  model: string;
  results: Record<string, QuestionResult>;
  usage: { latency_ms?: number; [key: string]: unknown };
}

/** Raised when the server returns a non-2xx response.
 *
 * `statusCode` is commonly:
 *   - 501: primitive/option-set not implemented by the current checkpoint
 *     (the expected outcome for anything but `choice` with
 *     options === ["entailment", "neutral", "contradiction"] today).
 *   - 422: malformed request (e.g. a `choice` question missing `options`).
 */
export class EkVachanAPIError extends Error {
  readonly statusCode: number;

  constructor(statusCode: number, message: string) {
    super(`ekVachan API error ${statusCode}: ${message}`);
    this.name = "EkVachanAPIError";
    this.statusCode = statusCode;
  }
}

export interface EkVachanClientOptions {
  /** Base URL of a self-hosted ekVachan server, e.g. "http://localhost:8000". */
  baseUrl: string;
  /** Request timeout in milliseconds. Default: 30000. */
  timeoutMs?: number;
  /** Custom fetch implementation (e.g. for Node < 18, or testing). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Minimal client for a self-hosted ekVachan server's `/v1/systemone`
 * endpoint. Fetch-based, no third-party dependency.
 *
 * @example
 * ```ts
 * const client = new EkVachanClient({ baseUrl: "http://localhost:8000" });
 * const response = await client.systemOne({
 *   state: "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.",
 *   questions: {
 *     nli: { type: "choice", options: ["entailment", "neutral", "contradiction"] },
 *   },
 * });
 * console.log(response.results.nli); // { choice: "entailment", probabilities: {...}, confidence: 0.98 }
 * ```
 *
 * Only this one question shape (`choice` over exactly
 * `["entailment", "neutral", "contradiction"]`) will succeed against
 * today's Phase 1 checkpoint; anything else rejects with
 * `EkVachanAPIError` (`statusCode === 501`).
 */
export class EkVachanClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: EkVachanClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  /**
   * POST /v1/systemone.
   *
   * Raises EkVachanAPIError on any non-2xx response — most commonly a 501
   * for `score`/`noul` or a `choice` option set the checkpoint wasn't
   * trained on (see the module docstring).
   */
  async systemOne(body: SystemOneRequestBody): Promise<SystemOneResponse> {
    return this.post<SystemOneResponse>("/v1/systemone", body);
  }

  /**
   * Convenience wrapper for the single most common case today: one
   * `choice` question. Returns just that question's result, e.g.
   * `{ choice: "entailment", probabilities: {...}, confidence: 0.98 }`.
   *
   * Only `options === ["entailment", "neutral", "contradiction"]` will
   * succeed against the Phase 1 checkpoint; any other option list raises
   * `EkVachanAPIError` with `statusCode === 501`.
   */
  async choice(args: {
    state: string;
    options: string[];
    instructions?: string;
    key?: string;
    model?: string;
  }): Promise<ChoiceResult> {
    const key = args.key ?? "choice";
    const response = await this.systemOne({
      state: args.state,
      model: args.model,
      questions: {
        [key]: { type: "choice", instructions: args.instructions, options: args.options },
      },
    });
    return response.results[key] as ChoiceResult;
  }

  /** GET /health. */
  async health(): Promise<{ status: string }> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/health`, {
        method: "GET",
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new EkVachanAPIError(res.status, await res.text());
      }
      return (await res.json()) as { status: string };
    } finally {
      clearTimeout(timer);
    }
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        let detail = text;
        try {
          const parsed = JSON.parse(text) as { detail?: string };
          detail = parsed.detail ?? text;
        } catch {
          // response body wasn't JSON; use the raw text as-is
        }
        throw new EkVachanAPIError(res.status, detail);
      }

      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }
}
