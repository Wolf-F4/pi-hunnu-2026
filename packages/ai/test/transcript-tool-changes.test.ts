import { Type } from "typebox";
import { describe, expect, test } from "vitest";
import { collapseSystemMessages, streamSimple } from "../src/compat.ts";
import type { Api, Context, Model, Tool } from "../src/types.ts";

class PayloadCaptured extends Error {}

function tool(name: string): Tool {
	return { name, description: `${name} tool`, parameters: Type.Object({}) };
}

async function capturePayload<T>(model: Model<Api>, context: Context): Promise<T> {
	let captured: T | undefined;
	const stream = streamSimple(model, context, {
		apiKey: "test-key",
		onPayload: (payload) => {
			captured = payload as T;
			throw new PayloadCaptured();
		},
	});
	await stream.result();
	if (!captured) throw new Error("Expected payload capture");
	return captured;
}

const baseTool = tool("base_tool");
const lateTool = tool("late_tool");
const lastTool = tool("last_tool");
const replacementTool = { ...baseTool, description: "replacement definition" };
const replacementContext: Context = {
	messages: [
		{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
		{ role: "user", content: "before", timestamp: 1 },
		{
			role: "system",
			content: "updated definition",
			toolsRemoved: [{ name: baseTool.name }],
			toolsAdded: [replacementTool],
			timestamp: 2,
		},
	],
};
const context: Context = {
	messages: [
		{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
		{ role: "user", content: "before", timestamp: 1 },
		{
			role: "system",
			content: "updated guidance",
			toolsRemoved: [{ name: "base_tool" }],
			toolsAdded: [lateTool],
			timestamp: 2,
		},
	],
};

describe("transcript tool changes", () => {
	test("uses a top-level prompt as the current snapshot while preserving tool transitions", async () => {
		const model: Model<"anthropic-messages"> = {
			id: "claude-sonnet-4-5",
			name: "Claude Sonnet 4.5",
			api: "anthropic-messages",
			provider: "anthropic",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
		};
		const collapsed = collapseSystemMessages(context, "current prompt");
		expect(collapsed.messages.filter((message) => message.role === "system")).toHaveLength(1);
		const initial = collapsed.messages[0];
		expect(initial?.role).toBe("system");
		if (initial?.role !== "system") throw new Error("expected initial system message");
		expect(initial.toolsAdded?.map((value) => value.name)).toEqual(["late_tool"]);

		const payload = await capturePayload<{
			system?: Array<{ text: string }>;
			tools?: Array<{ name: string }>;
			messages: Array<{ role: string; content: unknown }>;
		}>(model, collapsed);

		expect(payload.system?.map((block) => block.text)).toEqual(["current prompt"]);
		expect(payload.tools?.map((value) => value.name)).toEqual(["late_tool"]);
		expect(JSON.stringify(payload.messages)).not.toContain("base prompt");
		expect(JSON.stringify(payload.messages)).not.toContain("updated guidance");

		const clearedPayload = await capturePayload<{
			system?: Array<{ text: string }>;
			messages: Array<{ role: string; content: unknown }>;
		}>(model, collapseSystemMessages(context, ""));
		expect(clearedPayload.system).toBeUndefined();
		expect(JSON.stringify(clearedPayload.messages)).not.toContain("base prompt");
		expect(JSON.stringify(clearedPayload.messages)).not.toContain("updated guidance");
	});

	test("serializes Anthropic additions and removals in native system messages", async () => {
		const model: Model<"anthropic-messages"> = {
			id: "claude-opus-5",
			name: "Claude Opus 5",
			api: "anthropic-messages",
			provider: "anthropic",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsMidConvoSystemMessages: true, supportsMidConvoToolChanges: true },
		};
		const payload = await capturePayload<{
			betas?: string[];
			tools?: Array<{ name: string }>;
			messages: Array<{ role: string; content: Array<{ type: string; tool?: { name: string } }> }>;
		}>(model, context);

		expect(payload.betas).toContain("mid-conversation-tool-changes-2026-07-01");
		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.messages.at(-1)).toMatchObject({
			role: "system",
			content: [
				{ type: "text" },
				{ type: "tool_removal", tool: { name: "base_tool" } },
				{ type: "tool_addition", tool: { name: "late_tool" } },
			],
		});
	});

	test("anchors OpenAI additions at their system message", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsAdditionalTools: true },
		};
		const additionContext: Context = {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{ role: "system", content: "updated guidance", toolsAdded: [lateTool], timestamp: 2 },
				{ role: "user", content: "after", timestamp: 3 },
				{ role: "system", content: "more guidance", toolsAdded: [lastTool], timestamp: 4 },
			],
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; tools?: Array<{ name: string }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expect(
			payload.input
				.filter((item) => item.type === "additional_tools")
				.map((item) => item.tools?.map((value) => value.name)),
		).toEqual([["late_tool"], ["late_tool", "last_tool"]]);
	});

	test("maps system-message additions into synthetic tool search", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsToolSearch: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; tools?: Array<{ name: string }> }>;
		}>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{ role: "system", content: "updated guidance", toolsAdded: [lateTool], timestamp: 2 },
			],
		});

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expect(payload.input.map((item) => item.type)).toContain("tool_search_call");
		expect(
			payload.input.find((item) => item.type === "tool_search_output")?.tools?.map((value) => value.name),
		).toEqual(["late_tool"]);
	});

	test("uses the complete current tool definition instead of unsafe tool-search replacement", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsToolSearch: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string; description: string }>;
			input: Array<{ type?: string }>;
		}>(model, replacementContext);

		expect(payload.tools).toMatchObject([{ name: "base_tool", description: "replacement definition" }]);
		expect(payload.input.map((item) => item.type)).not.toContain("tool_search_call");
	});

	test("anchors Kimi additions in tool-bearing system messages", async () => {
		const model: Model<"openai-completions"> = {
			id: "kimi-k3",
			name: "Kimi K3",
			api: "openai-completions",
			provider: "moonshotai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsMidConvoToolAdditions: true },
		};
		const additionContext: Context = {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{ role: "system", content: "updated guidance", toolsAdded: [lateTool], timestamp: 2 },
			],
		};
		const payload = await capturePayload<{
			tools?: Array<{ function?: { name: string } }>;
			messages: Array<{ role: string; tools?: Array<{ function?: { name: string } }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool"]);
		expect(payload.messages.find((message) => message.tools)?.tools?.map((value) => value.function?.name)).toEqual([
			"late_tool",
		]);
	});

	test("uses the complete current tool definition instead of unsafe Kimi replacement", async () => {
		const model: Model<"openai-completions"> = {
			id: "kimi-k3",
			name: "Kimi K3",
			api: "openai-completions",
			provider: "moonshotai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsMidConvoToolAdditions: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ function?: { name: string; description: string } }>;
			messages: Array<{ role: string; tools?: unknown }>;
		}>(model, replacementContext);

		expect(payload.tools).toMatchObject([{ function: { name: "base_tool", description: "replacement definition" } }]);
		expect(payload.messages.some((message) => message.tools !== undefined)).toBe(false);
	});

	test("replaces OpenAI additional tool snapshots after removals", async () => {
		const model: Model<"openai-responses"> = {
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			baseUrl: "http://127.0.0.1:9",
			reasoning: true,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 100000,
			maxTokens: 1000,
			compat: { supportsAdditionalTools: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; tools?: Array<{ name: string }> }>;
		}>(model, context);

		expect(payload.tools).toBeUndefined();
		expect(
			payload.input
				.filter((item) => item.type === "additional_tools")
				.map((item) => item.tools?.map((value) => value.name)),
		).toEqual([["base_tool"], ["late_tool"]]);
	});
});
