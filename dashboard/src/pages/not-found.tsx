import { Link } from "react-router";
import { Compass } from "lucide-react";
import { buttonClass } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";

export function NotFoundPage() {
  return (
    <EmptyState
      icon={Compass}
      title="This page doesn't exist"
      action={
        <Link to="/overview" className={buttonClass("outline")}>
          Back to overview
        </Link>
      }
    />
  );
}
