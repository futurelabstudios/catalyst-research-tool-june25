import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  Activity,
  Info,
  Search,
  TextSearch,
  Brain,
  Pen,
  ChevronDown,
  ChevronUp,
  FileText, // For Internal KB
  CheckCircle2, // For Success
  XCircle, // For Failure
} from "lucide-react";
import { useEffect, useState, ReactNode } from "react";

export interface ProcessedEvent {
  title: string;
  data: any;
}

interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
}

/**
 * A helper function to determine the correct icon and title based on the event.
 * This is the "brain" of the component, making the timeline informative.
 */
function getEventVisuals(event: ProcessedEvent): {
  icon: ReactNode;
  title: string;
} {
  const eventTitle = event.title.toLowerCase();
  const eventData = (
    typeof event.data === "string" ? event.data : ""
  ).toLowerCase();

  // 1. Handle major lifecycle events first
  if (eventTitle.includes("planning")) {
    return {
      icon: <TextSearch className="h-4 w-4 text-sky-400" />,
      title: "Planning",
    };
  }
  if (eventTitle.includes("reflection")) {
    return {
      icon: <Brain className="h-4 w-4 text-purple-400" />,
      title: "Reflecting",
    };
  }
  if (eventTitle.includes("finalizing")) {
    return {
      icon: <Pen className="h-4 w-4 text-green-400" />,
      title: "Finalizing Answer",
    };
  }

  // 2. Handle granular "Research Progress" events by inspecting their data
  if (eventTitle.includes("research progress")) {
    if (eventData.includes("failed")) {
      return {
        icon: <XCircle className="h-4 w-4 text-red-400" />,
        title: "Search Failed",
      };
    }
    if (eventData.includes("finished search for:")) {
      return {
        icon: <CheckCircle2 className="h-4 w-4 text-green-400" />,
        title: "Web Search Complete",
      };
    }
    if (eventData.includes("finished retrieving from internal kb")) {
      return {
        icon: <CheckCircle2 className="h-4 w-4 text-green-400" />,
        title: "KB Retrieval Complete",
      };
    }
    if (eventData.includes("retrieving from internal kb")) {
      return {
        icon: <FileText className="h-4 w-4 text-neutral-400" />,
        title: "Searching Internal KB",
      };
    }
    if (eventData.includes("starting research with")) {
      return {
        icon: <Search className="h-4 w-4 text-blue-400" />,
        title: "Starting Research",
      };
    }
    // Default for any other web search-related progress
    return {
      icon: <Search className="h-4 w-4 text-blue-400" />,
      title: "Web Search in Progress",
    };
  }

  // 3. Fallback for any unknown event type
  return {
    icon: <Activity className="h-4 w-4 text-neutral-400" />,
    title: event.title,
  };
}

export function ActivityTimeline({
  processedEvents,
  isLoading,
}: ActivityTimelineProps) {
  const [isTimelineCollapsed, setIsTimelineCollapsed] =
    useState<boolean>(false);

  // Auto-collapse the timeline when the research is finished.
  useEffect(() => {
    if (!isLoading && processedEvents.length > 0) {
      setIsTimelineCollapsed(true);
    }
    if (isLoading) {
      setIsTimelineCollapsed(false);
    }
  }, [isLoading, processedEvents.length]);

  return (
    <Card className="border-none rounded-lg bg-neutral-700 max-h-fit">
      <CardHeader>
        <CardDescription className="flex items-center justify-between">
          <div
            className="flex items-center justify-start text-sm w-full cursor-pointer gap-2 text-neutral-100"
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
          >
            Agent Activity
            {isLoading && (
              <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />
            )}
            <span className="flex-grow" />{" "}
            {/* Pushes the chevron to the right */}
            {isTimelineCollapsed ? (
              <ChevronDown className="h-4 w-4 mr-2" />
            ) : (
              <ChevronUp className="h-4 w-4 mr-2" />
            )}
          </div>
        </CardDescription>
      </CardHeader>
      {!isTimelineCollapsed && (
        <ScrollArea className="max-h-96 overflow-y-auto" hideScrollbar={true}>
          <CardContent>
            {/* Initial loading state before any events */}
            {isLoading && processedEvents.length === 0 && (
              <div className="relative pl-8 pb-4">
                <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-800" />
                <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-800 flex items-center justify-center ring-4 ring-neutral-900">
                  <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                </div>
                <div>
                  <p className="text-sm text-neutral-300 font-medium">
                    Initializing agent...
                  </p>
                </div>
              </div>
            )}

            {processedEvents.length > 0 && (
              <div className="space-y-0">
                {processedEvents.map((eventItem, index) => {
                  const { icon, title } = getEventVisuals(eventItem);
                  return (
                    <div key={index} className="relative pl-8 pb-4">
                      {/* Timeline line */}
                      {index < processedEvents.length - 1 || isLoading ? (
                        <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-600" />
                      ) : null}

                      {/* Icon circle */}
                      <div className="absolute left-0.5 top-2 h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                        {icon}
                      </div>

                      {/* Text content */}
                      <div>
                        <p className="text-sm text-neutral-200 font-medium mb-0.5">
                          {title}
                        </p>
                        <p className="text-xs text-neutral-300 leading-relaxed">
                          {typeof eventItem.data === "string"
                            ? eventItem.data
                            : JSON.stringify(eventItem.data)}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Trailing loader when still loading after some events */}
            {isLoading && processedEvents.length > 0 && (
              <div className="relative pl-8 pb-4">
                <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                  <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                </div>
                <div>
                  <p className="text-sm text-neutral-300 font-medium">
                    Processing...
                  </p>
                </div>
              </div>
            )}

            {/* Idle state when not loading and no events */}
            {!isLoading && processedEvents.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-neutral-500 pt-10">
                <Info className="h-6 w-6 mb-3" />
                <p className="text-sm">No activity to display.</p>
                <p className="text-xs text-neutral-600 mt-1">
                  Activity will appear here during research.
                </p>
              </div>
            )}
          </CardContent>
        </ScrollArea>
      )}
    </Card>
  );
}
