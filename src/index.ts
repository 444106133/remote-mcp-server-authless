import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { z } from "zod";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Env {
  OPENROUTER_API_KEY: string;
}

interface ModelPricing {
  prompt?: string;
  completion?: string;
  image?: string;
  request?: string;
}

interface ModelArchitecture {
  input_modalities?: string[];
  output_modalities?: string[];
}

interface OpenRouterModel {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
  pricing?: ModelPricing;
  architecture?: ModelArchitecture;
  created?: number;
  top_provider?: unknown;
  supported_parameters?: unknown;
  default_parameters?: unknown;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models";

// In-memory cache (lives for the lifetime of the isolate)
let cachedData: { data: OpenRouterModel[] } | null = null;
let cacheTimestamp: number | null = null;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function extractProvider(modelId: string): string {
  const parts = modelId.split("/");
  return parts.length > 1 ? parts[0] : "unknown";
}

async function fetchModels(
  apiKey: string,
  forceRefresh = false
): Promise<{ data: OpenRouterModel[] }> {
  const now = Date.now();
  if (
    !forceRefresh &&
    cachedData &&
    cacheTimestamp &&
    now - cacheTimestamp < CACHE_TTL
  ) {
    return cachedData;
  }
  const res = await fetch(OPENROUTER_API_URL, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "HTTP-Referer": "https://mcp.workers.dev",
      "X-Title": "MCP OpenRouter Worker",
    },
  });
  if (!res.ok) throw new Error(`OpenRouter API error: ${res.status}`);
  const data = (await res.json()) as { data: OpenRouterModel[] };
  cachedData = data;
  cacheTimestamp = now;
  return data;
}

// ─── MCP Agent ────────────────────────────────────────────────────────────────

export class MyMCP extends McpAgent {
  server = new McpServer({
    name: "OpenRouter MCP",
    version: "1.0.0",
  });

  async init() {
    // ── list_models ──────────────────────────────────────────────────────────
    this.server.tool(
      "list_models",
      {
        provider: z.string().optional().describe("Filter by provider (e.g. anthropic, openai, google)"),
        category: z.string().optional().describe("Filter by category keyword in name/description"),
        max_price_per_1m_tokens: z.number().optional().describe("Max price per 1M input tokens (USD)"),
        min_context_length: z.number().optional().describe("Minimum context length required"),
        modality: z.enum(["text", "image", "audio", "video"]).optional().describe("Filter by input modality"),
        sort_by: z.enum(["price", "context_length", "name", "created"]).optional(),
        sort_order: z.enum(["asc", "desc"]).optional().default("asc"),
        limit: z.number().optional().default(50).describe("Max models to return (default 50, max 200)"),
        offset: z.number().optional().default(0).describe("Pagination offset"),
      },
      async (args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);
        let models = data.data;

        if (args.provider) {
          const p = args.provider.toLowerCase();
          models = models.filter((m) =>
            extractProvider(m.id).toLowerCase().includes(p)
          );
        }
        if (args.category) {
          const c = args.category.toLowerCase();
          models = models.filter(
            (m) =>
              m.name?.toLowerCase().includes(c) ||
              m.description?.toLowerCase().includes(c)
          );
        }
        if (args.max_price_per_1m_tokens !== undefined) {
          models = models.filter((m) => {
            const price = parseFloat(m.pricing?.prompt || "0") * 1_000_000;
            return price <= args.max_price_per_1m_tokens!;
          });
        }
        if (args.min_context_length !== undefined) {
          models = models.filter(
            (m) => (m.context_length || 0) >= args.min_context_length!
          );
        }
        if (args.modality) {
          models = models.filter((m) =>
            m.architecture?.input_modalities?.includes(args.modality!)
          );
        }

        if (args.sort_by) {
          const order = args.sort_order === "desc" ? -1 : 1;
          models.sort((a, b) => {
            if (args.sort_by === "price") {
              return (
                order *
                (parseFloat(a.pricing?.prompt || "0") -
                  parseFloat(b.pricing?.prompt || "0"))
              );
            }
            if (args.sort_by === "context_length") {
              return order * ((a.context_length || 0) - (b.context_length || 0));
            }
            if (args.sort_by === "name") {
              return order * (a.name || "").localeCompare(b.name || "");
            }
            if (args.sort_by === "created") {
              return order * ((a.created || 0) - (b.created || 0));
            }
            return 0;
          });
        }

        const totalCount = models.length;
        const limit = Math.min(args.limit ?? 50, 200);
        const offset = args.offset ?? 0;
        const paginated = models.slice(offset, offset + limit);
        const hasMore = offset + limit < totalCount;

        const formatted = paginated.map((m) => ({
          id: m.id,
          name: m.name,
          provider: extractProvider(m.id),
          description: m.description,
          context_length: m.context_length,
          pricing_per_1m_tokens: {
            input: (parseFloat(m.pricing?.prompt || "0") * 1_000_000).toFixed(2),
            output: (parseFloat(m.pricing?.completion || "0") * 1_000_000).toFixed(2),
          },
          modalities: {
            input: m.architecture?.input_modalities || [],
            output: m.architecture?.output_modalities || [],
          },
          created: m.created ? new Date(m.created * 1000).toISOString() : null,
        }));

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  total_count: totalCount,
                  count: formatted.length,
                  offset,
                  limit,
                  has_more: hasMore,
                  next_offset: hasMore ? offset + limit : null,
                  models: formatted,
                },
                null,
                2
              ),
            },
          ],
        };
      }
    );

    // ── get_model ────────────────────────────────────────────────────────────
    this.server.tool(
      "get_model",
      {
        model_id: z.string().describe("The model ID (e.g. anthropic/claude-3-opus)"),
      },
      async (args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);
        const model = data.data.find((m) => m.id === args.model_id);
        if (!model) throw new Error(`Model not found: ${args.model_id}`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  id: model.id,
                  name: model.name,
                  provider: extractProvider(model.id),
                  description: model.description,
                  created: model.created ? new Date(model.created * 1000).toISOString() : null,
                  architecture: model.architecture,
                  context_length: model.context_length,
                  pricing: {
                    prompt_per_token: model.pricing?.prompt,
                    completion_per_token: model.pricing?.completion,
                    image_per_token: model.pricing?.image,
                    request: model.pricing?.request,
                    per_1m_tokens: {
                      input: (parseFloat(model.pricing?.prompt || "0") * 1_000_000).toFixed(2),
                      output: (parseFloat(model.pricing?.completion || "0") * 1_000_000).toFixed(2),
                    },
                  },
                  top_provider: model.top_provider,
                  supported_parameters: model.supported_parameters,
                  default_parameters: model.default_parameters,
                },
                null,
                2
              ),
            },
          ],
        };
      }
    );

    // ── list_providers ───────────────────────────────────────────────────────
    this.server.tool(
      "list_providers",
      {},
      async (_args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);
        const providerMap: Record<string, { name: string; model_count: number; models: string[] }> = {};

        for (const model of data.data) {
          const p = extractProvider(model.id);
          if (!providerMap[p]) providerMap[p] = { name: p, model_count: 0, models: [] };
          providerMap[p].model_count++;
          providerMap[p].models.push(model.id);
        }

        const providers = Object.values(providerMap).sort(
          (a, b) => b.model_count - a.model_count
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ count: providers.length, providers }, null, 2),
            },
          ],
        };
      }
    );

    // ── compare_models ───────────────────────────────────────────────────────
    this.server.tool(
      "compare_models",
      {
        model_ids: z.array(z.string()).describe("Array of model IDs to compare"),
      },
      async (args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);

        const comparison = args.model_ids
          .map((id) => {
            const model = data.data.find((m) => m.id === id);
            if (!model) return null;
            return {
              id: model.id,
              name: model.name,
              provider: extractProvider(model.id),
              context_length: model.context_length,
              pricing_per_1m_tokens: {
                input: (parseFloat(model.pricing?.prompt || "0") * 1_000_000).toFixed(2),
                output: (parseFloat(model.pricing?.completion || "0") * 1_000_000).toFixed(2),
              },
              modalities: model.architecture?.input_modalities || [],
              supported_parameters: model.supported_parameters,
            };
          })
          .filter(Boolean);

        return {
          content: [
            { type: "text", text: JSON.stringify({ comparison }, null, 2) },
          ],
        };
      }
    );

    // ── search_models ────────────────────────────────────────────────────────
    this.server.tool(
      "search_models",
      {
        query: z.string().describe("Search query string"),
        search_in: z
          .enum(["name", "description", "both"])
          .optional()
          .default("both")
          .describe("Where to search"),
      },
      async (args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);
        const q = args.query.toLowerCase();
        const searchIn = args.search_in ?? "both";

        const results = data.data
          .filter((m) => {
            const byName =
              (searchIn === "name" || searchIn === "both") &&
              m.name?.toLowerCase().includes(q);
            const byDesc =
              (searchIn === "description" || searchIn === "both") &&
              m.description?.toLowerCase().includes(q);
            return byName || byDesc;
          })
          .map((m) => ({
            id: m.id,
            name: m.name,
            provider: extractProvider(m.id),
            description: m.description,
            context_length: m.context_length,
          }));

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { query: args.query, count: results.length, results },
                null,
                2
              ),
            },
          ],
        };
      }
    );

    // ── get_cheapest_models ──────────────────────────────────────────────────
    this.server.tool(
      "get_cheapest_models",
      {
        min_context_length: z.number().optional().describe("Minimum context length"),
        modality: z.string().optional().describe("Required input modality"),
        limit: z.number().optional().default(10).describe("Max results"),
      },
      async (args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        const data = await fetchModels(OPENROUTER_API_KEY);
        let models = data.data;

        if (args.min_context_length) {
          models = models.filter(
            (m) => (m.context_length || 0) >= args.min_context_length!
          );
        }
        if (args.modality) {
          models = models.filter((m) =>
            m.architecture?.input_modalities?.includes(args.modality!)
          );
        }

        models.sort(
          (a, b) =>
            parseFloat(a.pricing?.prompt || "0") -
            parseFloat(b.pricing?.prompt || "0")
        );
        models = models.slice(0, args.limit ?? 10);

        const formatted = models.map((m) => ({
          id: m.id,
          name: m.name,
          provider: extractProvider(m.id),
          context_length: m.context_length,
          pricing_per_1m_tokens: {
            input: (parseFloat(m.pricing?.prompt || "0") * 1_000_000).toFixed(2),
            output: (parseFloat(m.pricing?.completion || "0") * 1_000_000).toFixed(2),
          },
        }));

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ count: formatted.length, models: formatted }, null, 2),
            },
          ],
        };
      }
    );

    // ── refresh_cache ────────────────────────────────────────────────────────
    this.server.tool(
      "refresh_cache",
      {},
      async (_args, { env }) => {
        const { OPENROUTER_API_KEY } = env as unknown as Env;
        await fetchModels(OPENROUTER_API_KEY, true);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                status: "success",
                message: "Cache refreshed successfully",
                timestamp: new Date().toISOString(),
              }),
            },
          ],
        };
      }
    );
  }
}

// ─── Worker Entry Point ───────────────────────────────────────────────────────

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);
    if (url.pathname === "/mcp") {
      return MyMCP.serve("/mcp").fetch(request, env, ctx);
    }
    return new Response("Not found", { status: 404 });
  },
};
