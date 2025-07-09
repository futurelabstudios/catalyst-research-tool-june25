import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message as SdkMessage } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
// This is the cleanest way to extend the base type.
interface CustomAIMessage {
  type: "ai";
  id?: string;
  content: string;
  name?: string;
  tool_call_id?: string;
}
type Message = SdkMessage | CustomAIMessage;

// to check if a message is our custom type. TypeScript understands this.
function isCustomAIMessage(message: Message): message is CustomAIMessage {
  return message.type === "ai" && "tool_call_id" in message;
}

// Define the full state that the stream manages.
type StreamState = {
  messages: Message[];
  initial_search_query_count: number;
  max_research_loops: number;
  reasoning_model: string;
  use_web_search: boolean;
};

type OnUpdateEventHandler = (data: {
  [node: string]: Partial<StreamState>;
}) => void;

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const onUpdateEvent: OnUpdateEventHandler = (data) => {
    let processedEvent: ProcessedEvent | null = null;

    // The `data` object contains the output of the node that just ran.
    // Its key is the node's name (e.g., "research_step").
    const nodeName = Object.keys(data)[0];
    const nodeOutput = data[nodeName];

    // Check if the node that ran was 'research_step' and if it streamed a message.
    if (
      nodeName === "research_step" &&
      nodeOutput?.messages &&
      nodeOutput.messages.length > 0
    ) {
      // Get the last message *from the node's output*, not the component state.
      const lastMessage = nodeOutput.messages[nodeOutput.messages.length - 1];

      // Use the type guard to check if it's our special progress-update message.
      if (
        isCustomAIMessage(lastMessage) &&
        lastMessage.tool_call_id === "research_update"
      ) {
        processedEvent = {
          title: "Research Progress",
          data: lastMessage.content,
        };
      }
    }
    // Fallback to check for the major node completion events.
    else if (nodeName === "determine_tool_and_initial_action" && nodeOutput) {
      processedEvent = {
        title: "🤔 Planning Strategy",
        data: (nodeOutput as any).use_web_search
          ? "Decided to use Web Search. Generating queries..."
          : "Decided to use Internal KB. Selecting topic...",
      };
    } else if (nodeName === "reflection" && nodeOutput) {
      processedEvent = {
        title: "🧠 Reflection",
        data: (nodeOutput as any).is_sufficient
          ? "Research appears sufficient. Proceeding to finalize."
          : `Knowledge gap found: "${
              (nodeOutput as any).knowledge_gap
            }". Generating follow-up questions.`,
      };
    } else if (nodeName === "finalize_answer") {
      processedEvent = {
        title: "📝 Finalizing Answer",
        data: "Drafting the final report based on all findings.",
      };
      hasFinalizeEventOccurredRef.current = true;
    }

    if (processedEvent) {
      setProcessedEventsTimeline((prevEvents) => {
        if (
          prevEvents.length > 0 &&
          prevEvents[prevEvents.length - 1].data === processedEvent?.data
        ) {
          return prevEvents;
        }
        return [...prevEvents, processedEvent];
      });
    }
  };

  const { messages, submit, isLoading, stop } = useStream<StreamState>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : window.location.origin,
    assistantId: "agent",
    messagesKey: "messages",
    onUpdateEvent,
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !isLoading &&
      messages.length > 0
    ) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [messages, isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (
      submittedInputValue: string,
      effort: string,
      model: string,
      useWebSearch: boolean
    ) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 5;
          break;
      }

      const newMessages: SdkMessage[] = [
        ...(messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      submit({
        messages: newMessages,
        initial_search_query_count,
        max_research_loops,
        reasoning_model: model,
        use_web_search: useWebSearch,
      });
    },
    [messages, submit]
  );

  const handleCancel = useCallback(() => {
    stop();
  }, [stop]);

  const thread = { messages, submit, isLoading, stop };

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
        <div
          className={`flex-1 overflow-y-auto ${
            thread.messages.length === 0 ? "flex" : ""
          }`}
        >
          {thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
        </div>
      </main>
    </div>
  );
}
