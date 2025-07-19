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
  AlertTriangle,
  FileText,
  Zap,
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
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
    }

    // Handle loading states
    if (activity.status === "in_progress") {
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
    }

    if (activity.status === "failed") {
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    }

    // Handle specific icon types
    switch (activity.icon) {
      case "search":
        return <Search className="h-4 w-4 text-blue-500" />;
      case "read":
        return <FileText className="h-4 w-4 text-emerald-500" />;
      case "synthesize":
        return <Zap className="h-4 w-4 text-purple-500" />;
      case "web":
        return <Globe className="h-4 w-4 text-cyan-500" />;
      case "done":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />;
      case "error":
        return <AlertTriangle className="h-4 w-4 text-amber-500" />;
      case "globe":
        return <Globe className="h-4 w-4 text-cyan-500" />;
      case "database":
        return <Database className="h-4 w-4 text-indigo-500" />;
      case "brain":
        return <Brain className="h-4 w-4 text-purple-500" />;
      case "check-circle":
        return <CheckCircle className="h-4 w-4 text-emerald-500" />;
      default:
        break;
    }

    // Fallback to title-based icons (legacy support)
    const title = activity.title.toLowerCase();
    if (title.includes("generating") || title.includes("search")) {
      return <TextSearch className="h-4 w-4 text-blue-500" />;
    } else if (title.includes("thinking")) {
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
    } else if (title.includes("reflection")) {
      return <Brain className="h-4 w-4 text-purple-500" />;
    } else if (title.includes("research")) {
      return <Search className="h-4 w-4 text-blue-500" />;
    } else if (title.includes("reading") || title.includes("documents")) {
      return <FileText className="h-4 w-4 text-emerald-500" />;
    } else if (title.includes("finalizing") || title.includes("composed")) {
      return <Pen className="h-4 w-4 text-purple-500" />;
    }

    return <ActivityIcon className="h-4 w-4 text-slate-500" />;
  };

  const getStatusColor = (status: Activity["status"]) => {
    switch (status) {
      case "completed":
        return "bg-emerald-500";
      case "in_progress":
        return "bg-blue-500";
      case "failed":
        return "bg-red-500";
      case "skipped":
        return "bg-amber-500";
      default:
        return "bg-slate-500";
    }
  };

  const getImportanceIndicator = (importance: Activity["importance"]) => {
    switch (importance) {
      case "critical":
        return "ring-2 ring-blue-400/40";
      case "normal":
        return "ring-2 ring-slate-400/30";
      default:
        return "ring-2 ring-slate-400/20";
    }
  };

  const formatDetails = (activity: Activity) => {
    if (!activity.details) return null;

    // Handle reading completion messages
    if (
      activity.details.includes("Successfully read content for") &&
      activity.details.includes("out of")
    ) {
      const match = activity.details.match(
        /Successfully read content for (\d+) out of (\d+) files\.(?: Failed: (\d+)\.)?(?: Errors: (\d+)\.)?/
      );

      if (match) {
        const [, successful, total, failed = "0", errors = "0"] = match;
        const truncated =
          parseInt(total) - parseInt(successful) - parseInt(failed);

        return (
          <div className="space-y-1">
            <p className="text-xs text-slate-300">
              Successfully processed{" "}
              <span className="text-emerald-400 font-medium">
                {successful}/{total}
              </span>{" "}
              files
            </p>
            {parseInt(failed) > 0 && (
              <p className="text-xs text-red-400">
                • {failed} files failed to read
              </p>
            )}
            {truncated > 0 && (
              <p className="text-xs text-amber-400">
                • {truncated} files truncated due to size limits
              </p>
            )}
            {parseInt(errors) > 0 && (
              <p className="text-xs text-red-400">
                • {errors} processing errors encountered
              </p>
            )}
          </div>
        );
      }
    }

    return (
      <p className="text-xs text-slate-300 leading-relaxed">
        {activity.details}
      </p>
    );
  };

  useEffect(() => {
    if (!isLoading && activities.length !== 0) {
      setIsTimelineCollapsed(true);
    }
  }, [isLoading, activities]);

  return (
    <Card className="border border-slate-700/50 rounded-lg bg-neutral-900 backdrop-blur-sm shadow-lg">
      <CardHeader className="pb-3">
        <CardDescription className="flex items-center justify-between">
          <div
            className="flex items-center justify-start text-sm w-full cursor-pointer gap-2 text-slate-100 hover:text-slate-50 transition-colors"
            onClick={() => setIsTimelineCollapsed(!isTimelineCollapsed)}
          >
            <span className="font-medium">Research Progress</span>
            {isTimelineCollapsed ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronUp className="h-4 w-4" />
            )}
          </div>
        </CardDescription>
      </CardHeader>

      {!isTimelineCollapsed && (
        <ScrollArea className="max-h-96 overflow-y-auto">
          <CardContent className="pt-0">
            {isLoading && activities.length === 0 && (
              <div className="relative pl-10 pb-6">
                <div className="absolute left-4 top-4 h-full w-0.5 bg-gradient-to-b from-blue-500/50 to-transparent" />
                <div className="absolute left-2 top-2 h-6 w-6 rounded-full bg-slate-800 border-2 border-blue-500 flex items-center justify-center">
                  <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />
                </div>
                <div className="ml-1">
                  <p className="text-sm text-slate-200 font-medium">
                    Initializing search...
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Preparing to analyze your request
                  </p>
                </div>
              </div>
            )}

            {activities.length > 0 ? (
              <div className="space-y-0">
                {activities.map((activity, index) => (
                  <div
                    key={activity.id}
                    className="relative pl-10 pb-6 last:pb-2"
                  >
                    {index < activities.length - 1 ||
                    (isLoading && index === activities.length - 1) ? (
                      <div className="absolute left-4 top-6 h-full w-0.5 bg-gradient-to-b from-slate-600/80 to-slate-700/40" />
                    ) : null}

                    <div
                      className={`absolute left-2 top-2 h-6 w-6 rounded-full ${getStatusColor(
                        activity.status
                      )} flex items-center justify-center border-2 border-slate-800 ${getImportanceIndicator(
                        activity.importance
                      )}`}
                    >
                      {getEventIcon(activity, index)}
                    </div>

                    <div className="flex items-start justify-between ml-1">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm text-slate-100 font-medium">
                            {activity.title}
                          </p>
                          {activity.phase && (
                            <span className="text-xs text-slate-400 bg-slate-800 px-2 py-0.5 rounded-md border border-slate-700">
                              {activity.phase}
                            </span>
                          )}
                        </div>

                        {formatDetails(activity)}

                        {activity.progress && (
                          <div className="mt-2 flex items-center gap-3">
                            <div className="flex-1 bg-slate-800 rounded-full h-2 border border-slate-700">
                              <div
                                className="bg-gradient-to-r from-blue-500 to-blue-400 h-full rounded-full transition-all duration-500 ease-out"
                                style={{
                                  width: `${
                                    (activity.progress.current /
                                      activity.progress.total) *
                                    100
                                  }%`,
                                }}
                              />
                            </div>
                            <span className="text-xs text-slate-400 font-mono min-w-fit">
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
                          className="ml-3 h-7 w-7 p-0 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-700"
                        >
                          <RotateCcw className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && activities.length > 0 && (
                  <div className="relative pl-10 pb-4">
                    <div className="absolute left-2 top-2 h-6 w-6 rounded-full bg-slate-800 border-2 border-blue-500 flex items-center justify-center ring-2 ring-blue-400/30">
                      <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />
                    </div>
                    <div className="ml-1">
                      <p className="text-sm text-slate-200 font-medium">
                        Processing...
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        Analyzing and synthesizing information
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : !isLoading ? (
              <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                <div className="p-3 rounded-full bg-slate-800 mb-4">
                  <Info className="h-6 w-6" />
                </div>
                <p className="text-sm font-medium text-slate-400">
                  No research activity yet
                </p>
                <p className="text-xs text-slate-500 mt-1 text-center max-w-48">
                  Timeline will populate when processing begins
                </p>
              </div>
            ) : null}
          </CardContent>
        </ScrollArea>
      )}
    </Card>
  );
}
