import type { Context, SystemMessage, Tool, TranscriptContext } from "../types.ts";

export type { TranscriptContext } from "../types.ts";

/** Normalize the top-level initial state into a leading system message. */
export function normalizeContext(context: Context): TranscriptContext {
	const hasSystemPromptField = "systemPrompt" in context;
	const hasToolsField = "tools" in context;

	if (!hasSystemPromptField && !hasToolsField) {
		return context as TranscriptContext;
	}

	const systemPrompt = context.systemPrompt;
	const tools = context.tools;
	const hasSystemPrompt = systemPrompt !== undefined && systemPrompt.length > 0;
	const hasTools = tools !== undefined && tools.length > 0;

	if (!hasSystemPrompt && !hasTools) {
		return { messages: context.messages } as TranscriptContext;
	}

	const initialMessage: SystemMessage = {
		role: "system",
		content: systemPrompt ?? "",
		...(hasTools ? { toolsAdded: tools } : {}),
		timestamp: 0,
	};

	return { messages: [initialMessage, ...context.messages] } as TranscriptContext;
}

/**
 * Collapse prompt history for an API without native mid-conversation system messages.
 *
 * The caller supplies the exact current prompt because textual prompt updates cannot
 * be replayed mechanically. Tool changes are structured, so they are replayed into
 * the new leading message before intermediate system messages are removed.
 */
export function collapseSystemMessages(context: Context, systemPrompt: string): TranscriptContext {
	const normalizedContext = normalizeContext(context);
	const initialSystemMessage = getInitialSystemMessage(normalizedContext);
	return {
		messages: [
			{
				role: "system",
				content: systemPrompt,
				toolsAdded: getCurrentTools(normalizedContext),
				timestamp: initialSystemMessage?.timestamp ?? 0,
			},
			...normalizedContext.messages.filter((message) => message.role !== "system"),
		],
	} as TranscriptContext;
}

/** Return the leading system message, if the transcript starts with one. */
export function getInitialSystemMessage(context: TranscriptContext): SystemMessage | undefined {
	const first = context.messages[0];
	return first?.role === "system" ? first : undefined;
}

/** Return tools declared by the leading system message. */
export function getInitialTools(context: TranscriptContext): Tool[] {
	return getInitialSystemMessage(context)?.toolsAdded ?? [];
}

/** Resolve the tools available after applying every transcript delta in order. */
export function getCurrentTools(context: TranscriptContext): Tool[] {
	const tools = new Map<string, Tool>();
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsRemoved ?? []) tools.delete(tool.name);
		for (const tool of message.toolsAdded ?? []) tools.set(tool.name, tool);
	}
	return [...tools.values()];
}

/** Return a context without its leading system message. */
export function withoutInitialSystemMessage(context: TranscriptContext): TranscriptContext {
	return getInitialSystemMessage(context) ? ({ messages: context.messages.slice(1) } as TranscriptContext) : context;
}
