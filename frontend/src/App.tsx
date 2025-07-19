import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Activity } from "./lib/activity-types";

const MAX_MESSAGE_LENGTH = 20000; // Approx. 20KB

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
    reasoning_model: string;
    use_web_search: boolean;
    activity_feed?: Activity[]; // New: expecting structured activities from backend
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "https://catalyst-research-tool-june25-production.up.railway.app",
    assistantId: "agent",
    messagesKey: "messages",
    onFinish: (event: any) => {
      console.log(event);
    },
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
      effort: string,
      model: string,
      useWebSearch: boolean
    ) => {
      if (!submittedInputValue.trim()) return;

      // Clear current activities for new request
      setCurrentActivities([]);
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
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];

      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
        use_web_search: useWebSearch,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  // NEW: Retry function for failed activities
  const handleRetryActivity = useCallback((activityId: string) => {
    setCurrentActivities((prevActivities) =>
      prevActivities.map((activity) =>
        activity.id === activityId
          ? { ...activity, status: "in_progress" as const }
          : activity
      )
    );
    // Here you would typically send a retry request to the backend
    // For now, we'll just update the UI state
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
              // NEW: Pass structured activities instead of simple events
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
