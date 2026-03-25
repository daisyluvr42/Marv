export type GoalFrame = {
  objective: string;
  successCriteria: string[];
  constraints: string[];
  complexity: "trivial" | "moderate" | "complex";
  goalType: "inquiry" | "mutation";
};
