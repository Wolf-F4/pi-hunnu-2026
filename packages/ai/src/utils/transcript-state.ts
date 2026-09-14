import type { Api, Model, Tool, ToolReference, TranscriptContext } from "../types.ts";
import { getCurrentTools, getInitialTools } from "./normalize-context.ts";

export interface TranscriptCapabilities {
	midConversationSystemMessages: boolean;
}

export interface ToolStateChanges {
	toolsAdded: Tool[];
	toolsRemoved: ToolReference[];
}

export interface ResponsesTranscriptToolState {
	requestTools: Tool[];
	additionalToolsIncludeInitial: boolean;
}

type TranscriptCompat = {
	supportsMidConvoSystemMessages?: boolean;
	supportsMidConvoToolAdditions?: boolean;
	supportsAdditionalTools?: boolean;
	supportsToolSearch?: boolean;
};

const RESPONSES_APIS = new Set<Api>(["openai-responses", "azure-openai-responses", "openai-codex-responses"]);

/** Capabilities that preserve transcript position without rewriting the initial provider state. */
export function getTranscriptCapabilities(model: Model<Api>): TranscriptCapabilities {
	const compat = model.compat as TranscriptCompat | undefined;
	const nativeTranscript = model.api === "pi-messages" || model.api.startsWith("faux:");
	return {
		midConversationSystemMessages:
			nativeTranscript ||
			RESPONSES_APIS.has(model.api) ||
			((model.api === "anthropic-messages" || model.api === "openai-completions") &&
				compat?.supportsMidConvoSystemMessages === true) ||
			model.api === "mistral-conversations",
	};
}

/** Strip executable and display-only fields from a tool before transcript comparison or persistence. */
export function toToolDeclaration(tool: Tool): Tool {
	return {
		name: tool.name,
		description: tool.description,
		parameters: JSON.parse(JSON.stringify(tool.parameters)) as unknown as Tool["parameters"],
		...(tool.constrainedSampling === undefined ? {} : { constrainedSampling: tool.constrainedSampling }),
	};
}

function declarationsEqual(left: Tool, right: Tool): boolean {
	return JSON.stringify(toToolDeclaration(left)) === JSON.stringify(toToolDeclaration(right));
}

/** Compare two complete tool states, treating changed definitions as removal followed by addition. */
export function getToolStateChanges(previous: readonly Tool[], current: readonly Tool[]): ToolStateChanges {
	const previousTools = new Map(previous.map((tool) => [tool.name, tool]));
	const currentTools = new Map(current.map((tool) => [tool.name, tool]));
	return {
		toolsAdded: current
			.filter((tool) => {
				const previousTool = previousTools.get(tool.name);
				return previousTool === undefined || !declarationsEqual(previousTool, tool);
			})
			.map(toToolDeclaration),
		toolsRemoved: previous
			.filter((tool) => {
				const currentTool = currentTools.get(tool.name);
				return currentTool === undefined || !declarationsEqual(tool, currentTool);
			})
			.map((tool) => ({ name: tool.name })),
	};
}

/** Every definition referenced by transcript tool state, in first-declaration order. */
export function getDeclaredTools(context: TranscriptContext): Tool[] {
	const definitions = new Map<string, Tool>();
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) definitions.set(tool.name, tool);
	}
	return [...definitions.values()];
}

/** Whether tool history contains a removal or same-name replacement that an addition-only transport cannot replay. */
export function hasNonAdditiveToolChanges(context: TranscriptContext): boolean {
	const declaredNames = new Set<string>();
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		if ((message.toolsRemoved?.length ?? 0) > 0) return true;
		for (const tool of message.toolsAdded ?? []) {
			if (declaredNames.has(tool.name)) return true;
			declaredNames.add(tool.name);
		}
	}
	return false;
}

/** Whether this request needs the lossy current-state projection instead of transcript system history. */
export function shouldCollapseSystemMessages(model: Model<Api>, context: TranscriptContext): boolean {
	if (!getTranscriptCapabilities(model).midConversationSystemMessages) return true;
	if (!hasNonAdditiveToolChanges(context)) return false;
	const compat = model.compat as TranscriptCompat | undefined;
	if (model.api === "openai-completions") return compat?.supportsMidConvoToolAdditions === true;
	return (
		RESPONSES_APIS.has(model.api) && compat?.supportsToolSearch === true && compat.supportsAdditionalTools !== true
	);
}

/** Resolve the shared OpenAI Responses tool-loading policy for one transcript. */
export function getResponsesTranscriptToolState(
	context: TranscriptContext,
	supportsAdditionalTools: boolean,
	supportsToolSearch: boolean,
): ResponsesTranscriptToolState {
	const hasNonAdditiveChanges = hasNonAdditiveToolChanges(context);
	const additionalToolsIncludeInitial = supportsAdditionalTools && hasNonAdditiveChanges;
	return {
		additionalToolsIncludeInitial,
		requestTools: supportsAdditionalTools
			? additionalToolsIncludeInitial
				? []
				: getInitialTools(context)
			: supportsToolSearch && !hasNonAdditiveChanges
				? getInitialTools(context)
				: getCurrentTools(context),
	};
}
