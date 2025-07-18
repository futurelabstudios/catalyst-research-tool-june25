import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  Activity as ActivityIcon,
  Info,
  Search,
  TextSearch,
  Brain,
  Pen,
  ChevronDown,
  ChevronUp,
  Globe,
  Database,
  CheckCircle,
  AlertCircle,
  RotateCcw,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Activity } from "@/lib/activity-types";

interface ActivityTimelineProps {
  activities: Activity[];
  isLoading: boolean;
  onRetryActivity?: (activityId: string) => void;
}

export function ActivityTimeline({
  activities,
  isLoading,
  onRetryActivity,
}: ActivityTimelineProps) {
  const [isTimelineCollapsed, setIsTimelineCollapsed] =
    useState<boolean>(false);

  const getEventIcon = (activity: Activity, index: number) => {
    if (index === 0 && isLoading && activities.length === 0) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    }

    // Handle loading states
    if (activity.status === "in_progress") {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    }

    if (activity.status === "failed") {
      return <AlertCircle className="h-4 w-4 text-red-400" />;
    }

    // Handle specific icon types
    switch (activity.icon) {
      case "search":
        return <Search className="h-4 w-4 text-neutral-400" />;
      case "globe":
        return <Globe className="h-4 w-4 text-neutral-400" />;
      case "database":
        return <Database className="h-4 w-4 text-neutral-400" />;
      case "brain":
        return <Brain className="h-4 w-4 text-neutral-400" />;
      case "check-circle":
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      default:
        break;
    }

    // Fallback to title-based icons (legacy support)
    const title = activity.title.toLowerCase();
    if (title.includes("generating") || title.includes("search")) {
      return <TextSearch className="h-4 w-4 text-neutral-400" />;
    } else if (title.includes("thinking")) {
      return <Loader2 className="h-4 w-4 text-neutral-400 animate-spin" />;
    } else if (title.includes("reflection")) {
      return <Brain className="h-4 w-4 text-neutral-400" />;
    } else if (title.includes("research")) {
      return <Search className="h-4 w-4 text-neutral-400" />;
    } else if (title.includes("finalizing")) {
      return <Pen className="h-4 w-4 text-neutral-400" />;
    }

    return <ActivityIcon className="h-4 w-4 text-neutral-400" />;
  };

  const getStatusColor = (status: Activity["status"]) => {
    switch (status) {
      case "completed":
        return "bg-green-600";
      case "in_progress":
        return "bg-blue-600";
      case "failed":
        return "bg-red-600";
      case "skipped":
        return "bg-neutral-600";
      default:
        return "bg-neutral-600";
    }
  };

  const getImportanceIndicator = (importance: Activity["importance"]) => {
    switch (importance) {
      case "critical":
        return "ring-4 ring-blue-500/30";
      case "normal":
        return "ring-2 ring-blue-500/20";
      default:
        return "ring-4 ring-neutral-700";
    }
  };

  useEffect(() => {
    if (!isLoading && activities.length !== 0) {
      setIsTimelineCollapsed(true);
    }
  }, [isLoading, activities]);

  return (
    <Card className="border-none rounded-lg bg-neutral-700 max-h-fit">
      <CardHeader>
        <CardDescription className="flex items-center justify-between">
          <div
            className="flex items-center justify-start text-sm w-full cursor-pointer gap-2 text-neutral-100"
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
          >
            Research
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
            {isLoading && activities.length === 0 && (
              <div className="relative pl-8 pb-4">
                <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-800" />
                <div className="absolute left-0.5 top-2 h-5 w-5 rounded-full bg-neutral-800 flex items-center justify-center ring-4 ring-neutral-900">
                  <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                </div>
                <div>
                  <p className="text-sm text-neutral-300 font-medium">
                    Searching...
                  </p>
                </div>
              </div>
            )}
            {activities.length > 0 ? (
              <div className="space-y-0">
                {activities.map((activity, index) => (
                  <div key={activity.id} className="relative pl-8 pb-4">
                    {index < activities.length - 1 ||
                    (isLoading && index === activities.length - 1) ? (
                      <div className="absolute left-3 top-3.5 h-full w-0.5 bg-neutral-600" />
                    ) : null}
                    <div
                      className={`absolute left-0.5 top-2 h-6 w-6 rounded-full ${getStatusColor(
                        activity.status
                      )} flex items-center justify-center ${getImportanceIndicator(
                        activity.importance
                      )}`}
                    >
                      {getEventIcon(activity, index)}
                    </div>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-0.5">
                          <p className="text-sm text-neutral-200 font-medium">
                            {activity.title}
                          </p>
                          {activity.phase && (
                            <span className="text-xs text-neutral-400 bg-neutral-800 px-2 py-0.5 rounded">
                              {activity.phase}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-neutral-300 leading-relaxed">
                          {activity.details}
                        </p>
                        {activity.progress && (
                          <div className="mt-1 flex items-center gap-2">
                            <div className="flex-1 bg-neutral-800 rounded-full h-1.5">
                              <div
                                className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                                style={{
                                  width: `${
                                    (activity.progress.current /
                                      activity.progress.total) *
                                    100
                                  }%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-neutral-400">
                              {activity.progress.current}/
                              {activity.progress.total}
                            </span>
                          </div>
                        )}
                      </div>
                      {activity.status === "failed" && onRetryActivity && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onRetryActivity(activity.id)}
                          className="ml-2 h-6 w-6 p-0 text-neutral-400 hover:text-neutral-200"
                        >
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
                {isLoading && activities.length > 0 && (
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
              </div>
            ) : !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full text-neutral-500 pt-10">
                <Info className="h-6 w-6 mb-3" />
                <p className="text-sm">No activity to display.</p>
                <p className="text-xs text-neutral-600 mt-1">
                  Timeline will update during processing.
                </p>
              </div>
            ) : null}
          </CardContent>
        </ScrollArea>
      )}
    </Card>
  );
}
