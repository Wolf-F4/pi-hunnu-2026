import type { Api, Model, Tool, TranscriptContext } from "../types.ts";

export interface TranscriptCapabilities {
	midConversationSystemMessages: boolean;
}

type TranscriptCompat = {
	supportsMidConvoSystemMessages?: boolean;
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
			(model.api === "anthropic-messages" && compat?.supportsMidConvoSystemMessages === true) ||
			model.api === "openai-completions" ||
			model.api === "mistral-conversations",
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

/** Whether a tool name is declared again later, requiring replacement semantics. */
export function hasToolDefinitionReplacements(context: TranscriptContext): boolean {
	const declaredNames = new Set<string>();
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) {
			if (declaredNames.has(tool.name)) return true;
			declaredNames.add(tool.name);
		}
	}
	return false;
}
