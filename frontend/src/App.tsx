import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Activity } from "./lib/activity-types";

export default function App() {
  const [currentActivities, setCurrentActivities] = useState<Activity[]>([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, Activity[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);

  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    index_search_model: string;
    answer_model: string;
    reflection_model: string;
    query_generator_model: string;
    use_web_search: boolean;
    activity_feed?: Activity[];
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "https://catalyst-research-tool-june25-production.up.railway.app",
    assistantId: "agent",
    messagesKey: "messages",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onFinish: (event: any) => {
      console.log(event);
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onUpdateEvent: (event: any) => {
      console.log(event);
      let activityFeed = null;
      if (event.search_kb_index?.activity_feed) {
        activityFeed = event.search_kb_index.activity_feed;
      } else if (event.retrieve_kb_content?.activity_feed) {
        activityFeed = event.retrieve_kb_content.activity_feed;
      } else if (event.finalize_answer?.activity_feed) {
        activityFeed = event.finalize_answer.activity_feed;
        hasFinalizeEventOccurredRef.current = true;
      } else if (event.activity_feed) {
        activityFeed = event.activity_feed;
      }

      if (activityFeed && Array.isArray(activityFeed)) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const newActivities = activityFeed.map((activity: any) => ({
          ...activity,
          timestamp: activity.timestamp
            ? new Date(activity.timestamp)
            : new Date(),
        }));

        setCurrentActivities((prevActivities) => {
          const activityMap = new Map(prevActivities.map((a) => [a.id, a]));
          newActivities.forEach((newActivity: Activity) => {
            if (activityMap.has(newActivity.id)) {
              activityMap.set(newActivity.id, {
                ...activityMap.get(newActivity.id)!,
                ...newActivity,
                timestamp: activityMap.get(newActivity.id)!.timestamp,
              });
            } else {
              activityMap.set(newActivity.id, newActivity);
            }
          });
          return Array.from(activityMap.values()).sort(
            (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
          );
        });
      }
    },
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
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...currentActivities],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, currentActivities]);

  const handleSubmit = useCallback(
    (
      submittedInputValue: string,
      mode: string, // Updated parameter
      useWebSearch: boolean
    ) => {
      if (!submittedInputValue.trim()) return;

      setCurrentActivities([]);
      hasFinalizeEventOccurredRef.current = false;

      // Define models based on the selected mode
      let index_search_model = "";
      let answer_model = "";
      let reflection_model = "";
      let query_generator_model = "";

      if (mode === "fast") {
        index_search_model = "gemini-2.5-flash-lite-preview-06-17";
        answer_model = "gemini-2.5-flash-lite-preview-06-17";
        reflection_model = "gemini-2.5-flash-lite-preview-06-17";
        query_generator_model = "gemini-2.5-flash-lite-preview-06-17";
      } else {
        index_search_model = "gemini-2.5-flash";
        answer_model = "gemini-2.5-flash";
        reflection_model = "gemini-2.5-flash";
        query_generator_model = "gemini-2.5-flash";
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];

      // Submit with the new dynamic configuration
      thread.submit({
        messages: newMessages,
        initial_search_query_count: 3,
        max_research_loops: 3,
        index_search_model,
        answer_model,
        reflection_model,
        query_generator_model,
        use_web_search: useWebSearch,
      });
    },
    [thread]
  );
  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  const handleRetryActivity = useCallback((activityId: string) => {
    setCurrentActivities((prevActivities) =>
      prevActivities.map((activity) =>
        activity.id === activityId
          ? { ...activity, status: "in_progress" as const }
          : activity
      )
    );
  }, []);

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
              liveActivities={currentActivities}
              historicalActivities={historicalActivities}
              onRetryActivity={handleRetryActivity}
            />
          )}
        </div>
      </main>
    </div>
  );
}
