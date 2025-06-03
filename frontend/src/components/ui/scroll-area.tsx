import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";

import { cn } from "@/lib/utils";

function ScrollArea({
  className,
  children,
  hideScrollbar = false, // New prop, defaults to false
  viewportClassName,
  ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.Root> & {
  hideScrollbar?: boolean;
  viewportClassName?: string;
}) {
  return (
    <ScrollAreaPrimitive.Root
      data-slot="scroll-area"
      // Add overflow-hidden to the root to help ensure no native scrollbars appear
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        data-slot="scroll-area-viewport"
        // Removed focus styles from viewport as scrollbar is meant to be hidden
        // Added scrollbar-hide utility class for robustness across browsers
        className={cn(
          "size-full rounded-[inherit] scrollbar-hide",
          viewportClassName
        )}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      {!hideScrollbar && <ScrollBar />} {/* Conditionally render ScrollBar */}
      {!hideScrollbar && <ScrollAreaPrimitive.Corner />}{" "}
      {/* Conditionally render Corner */}
    </ScrollAreaPrimitive.Root>
  );
}

// ScrollBar component remains largely the same, it's just conditionally rendered.
// If you wanted to always render it but make it invisible, you'd style it here.
function ScrollBar({
  className,
  orientation = "vertical",
  ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>) {
  return (
    <ScrollAreaPrimitive.ScrollAreaScrollbar
      data-slot="scroll-area-scrollbar"
      orientation={orientation}
      className={cn(
        "flex touch-none p-px transition-colors select-none",
        orientation === "vertical" &&
          "h-full w-2.5 border-l border-l-transparent",
        orientation === "horizontal" &&
          "h-2.5 flex-col border-t border-t-transparent",
        "data-[state=hidden]:hidden", // Added this in case Radix sets this state
        className
      )}
      {...props}
    >
      <ScrollAreaPrimitive.ScrollAreaThumb
        data-slot="scroll-area-thumb"
        className="bg-border relative flex-1 rounded-full"
      />
    </ScrollAreaPrimitive.ScrollAreaScrollbar>
  );
}

export { ScrollArea, ScrollBar };
